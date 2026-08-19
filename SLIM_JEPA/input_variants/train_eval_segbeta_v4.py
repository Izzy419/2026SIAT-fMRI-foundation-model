# -*- coding: utf-8 -*-
"""
train_eval_segbeta_v4.py
方案3 v4 projector训练 + validation model selection + 100-way/1-way test。

关键修复:
1. 使用 feats3v4_train/test.npy，避免旧版文件混用。
2. 同一刺激(图像)对应多个trial时，采用multi-positive InfoNCE，
   同图target之间不会互相作为负样本。
3. 从训练集按image-level划分validation，防止同图trial泄漏到train/val。
4. 每10 epoch按validation Top-1保存best；最后重新加载best checkpoint做test。
5. 测试支持完整27000试次，也可 --eval-n 800做快速检查。

用法:
    CUDA_VISIBLE_DEVICES=1 python train_eval_segbeta_v4.py --eval-n 800
    CUDA_VISIBLE_DEVICES=1 nohup python train_eval_segbeta_v4.py --eval-n 27000 > segbeta_v4.log 2>&1 &
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import argparse
import time
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from config import DATA_DIR

DATA = str(DATA_DIR)
EPOCHS = 150
LR = 3e-4
BATCH = 128
WARMUP_EPOCHS = 5
TEMP = 0.1
VAL_EVERY = 10
VAL_FRACTION = 0.1
SEED = 0

parser = argparse.ArgumentParser()
parser.add_argument('--prefix', default='feats3v4')
parser.add_argument('--eval-n', type=int, default=27000)
parser.add_argument('--epochs', type=int, default=EPOCHS)
parser.add_argument('--val-every', type=int, default=VAL_EVERY)
parser.add_argument('--batch', type=int, default=BATCH)
args = parser.parse_args()

np.random.seed(SEED)
torch.manual_seed(SEED)

train_feat_path = f'{DATA}/{args.prefix}_train.npy'
test_feat_path = f'{DATA}/{args.prefix}_test.npy'
ckpt_path = f'{DATA}/projector_{args.prefix}_v4.pth'
result_path = f'{DATA}/weekend_results.txt'

for p in (train_feat_path, test_feat_path):
    if not os.path.exists(p):
        raise FileNotFoundError(p)

feats_all = np.load(train_feat_path).astype(np.float32, copy=False)
feats_te = np.load(test_feat_path).astype(np.float32, copy=False)
train_idx = np.load(f'{DATA}/train_idx.npy')
test_idx = np.load(f'{DATA}/test_idx.npy')
trial_imgidx = np.load(f'{DATA}/trial_imgidx.npy')
clip_emb = np.load(f'{DATA}/clip_emb.npy').astype(np.float32, copy=False)
clip_uidx = np.load(f'{DATA}/clip_unique_idx.npy')
id2pos = {int(c): p for p, c in enumerate(clip_uidx)}

if feats_all.shape[0] != len(train_idx):
    raise ValueError(f'train feature rows {feats_all.shape[0]} != train_idx {len(train_idx)}')
if feats_te.shape[0] != len(test_idx):
    raise ValueError(f'test feature rows {feats_te.shape[0]} != test_idx {len(test_idx)}')
if feats_all.shape[1] != feats_te.shape[1]:
    raise ValueError('train/test feature dimension不一致')
if not np.isfinite(feats_all).all() or not np.isfinite(feats_te).all():
    raise ValueError('feature中存在NaN/Inf')

clip_tr = np.stack([clip_emb[id2pos[int(trial_imgidx[t])]] for t in train_idx])
clip_te = np.stack([clip_emb[id2pos[int(trial_imgidx[t])]] for t in test_idx])
clip_tr = clip_tr / (np.linalg.norm(clip_tr, axis=1, keepdims=True) + 1e-9)
clip_te = clip_te / (np.linalg.norm(clip_te, axis=1, keepdims=True) + 1e-9)

# -------- image-level train/val split --------
train_img_ids = np.array([int(trial_imgidx[t]) for t in train_idx])
unique_imgs = np.unique(train_img_ids)
rng = np.random.default_rng(SEED)
rng.shuffle(unique_imgs)
n_val_imgs = max(1, int(round(len(unique_imgs) * VAL_FRACTION)))
val_img_ids = set(unique_imgs[:n_val_imgs].tolist())
val_mask = np.array([x in val_img_ids for x in train_img_ids])
tr_mask = ~val_mask

Xtr, Ytr, Itr = feats_all[tr_mask], clip_tr[tr_mask], train_img_ids[tr_mask]
Xva, Yva, Iva = feats_all[val_mask], clip_tr[val_mask], train_img_ids[val_mask]

print(f'特征: train={feats_all.shape}, test={feats_te.shape}', flush=True)
print(f'image-level split: train={len(Xtr)} trials / {len(np.unique(Itr))} imgs, '
      f'val={len(Xva)} trials / {len(np.unique(Iva))} imgs', flush=True)

class ClipProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.GELU(),
            nn.Linear(in_dim, in_dim), nn.GELU(),
            nn.Linear(in_dim, out_dim)
        )
    def forward(self, x):
        return F.normalize(self.mlp(x), dim=-1)

class PairedSet(Dataset):
    def __init__(self, x, y, img):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.img = torch.from_numpy(img.astype(np.int64))
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i], self.img[i]


def multi_positive_infonce(q, k, image_ids, temp=TEMP):
    """每个query可有多个正key；同image的其它trial不作为负样本。"""
    logits = q @ k.T / temp
    pos = image_ids[:, None] == image_ids[None, :]
    lse_all = torch.logsumexp(logits, dim=1)
    neg_inf = torch.finfo(logits.dtype).min
    pos_logits = logits.masked_fill(~pos, neg_inf)
    lse_pos = torch.logsumexp(pos_logits, dim=1)
    return (lse_all - lse_pos).mean()


model = ClipProjector(feats_all.shape[1], clip_tr.shape[1]).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
train_dl = DataLoader(PairedSet(Xtr, Ytr, Itr), batch_size=args.batch, shuffle=True, drop_last=False)

steps_per_epoch = len(train_dl)
total_steps = args.epochs * steps_per_epoch
warm_steps = WARMUP_EPOCHS * steps_per_epoch
sched = torch.optim.lr_scheduler.LambdaLR(
    opt,
    lambda s: s / max(1, warm_steps) if s < warm_steps else
    0.5 * (1 + np.cos(np.pi * (s - warm_steps) / max(1, total_steps - warm_steps)))
)


def retrieval_eval(model, feats, img_pos, clip_n, candidate_n=99, seed=0):
    model.eval()
    rng = np.random.default_rng(seed)
    test_unique = np.unique(img_pos)
    t1 = t5 = t10 = one = 0
    with torch.no_grad():
        for i in range(len(feats)):
            q = model(torch.from_numpy(feats[i:i+1]).cuda()).cpu().numpy()[0]
            gt_pos = img_pos[i]
            gt = clip_n[gt_pos]
            others = test_unique[test_unique != gt_pos]
            n = min(candidate_n, len(others))
            dist = rng.choice(others, size=n, replace=False)
            pool = np.concatenate([gt[None], clip_n[dist]], axis=0)
            rank = int(np.argsort(-(pool @ q))[0])
            t1 += rank == 0
            t5 += rank < 5
            t10 += rank < 10
            d1 = rng.choice(others, size=1, replace=False)[0]
            one += int(np.argmax(np.array([gt, clip_n[d1]]) @ q) == 0)
    n = len(feats)
    return t1/n, t5/n, t10/n, one/n

# Validation的img_pos直接使用全局clip index
val_img_pos = np.array([id2pos[x] for x in Iva], dtype=np.int64)
clip_n = clip_emb / (np.linalg.norm(clip_emb, axis=1, keepdims=True) + 1e-9)

best_val = -1.0
best_epoch = -1
t0 = time.time()
for ep in range(args.epochs):
    model.train()
    total_loss = 0.0
    for xb, yb, ib in train_dl:
        xb = xb.cuda(non_blocking=True)
        yb = F.normalize(yb.cuda(non_blocking=True), dim=-1)
        ib = ib.cuda(non_blocking=True)
        q = model(xb)
        loss = multi_positive_infonce(q, yb, ib)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        total_loss += float(loss.item())

    avg = total_loss / max(1, len(train_dl))
    print(f'epoch {ep+1:03d}: train multi-pos loss={avg:.4f}, lr={opt.param_groups[0]["lr"]:.2e}', flush=True)

    if ((ep + 1) % args.val_every == 0) or ep == args.epochs - 1:
        va1, va5, va10, vaone = retrieval_eval(model, Xva, val_img_pos, clip_n, seed=SEED)
        print(f'  [VAL] Top-1={va1:.4f} Top-5={va5:.4f} Top-10={va10:.4f} 1-way={vaone:.4f}', flush=True)
        if va1 > best_val:
            best_val = va1
            best_epoch = ep + 1
            torch.save({
                'model': model.state_dict(),
                'epoch': best_epoch,
                'val_top1': float(va1),
                'prefix': args.prefix,
            }, ckpt_path)
            print(f'  [保存best] {ckpt_path} (VAL Top-1={va1:.4f})', flush=True)

# -------- reload best, final test --------
ckpt = torch.load(ckpt_path, map_location='cuda')
model.load_state_dict(ckpt['model'])
print(f'加载best checkpoint: epoch={ckpt["epoch"]}, val_top1={ckpt["val_top1"]:.4f}', flush=True)

test_img_pos = np.array([id2pos[int(trial_imgidx[t])] for t in test_idx], dtype=np.int64)
n = min(args.eval_n, len(feats_te))

a1, a5, a10, w1 = retrieval_eval(model, feats_te[:n], test_img_pos[:n], clip_n, seed=SEED)
line = (f'[方案3v4-FIR+GLM-multipos] N={n} '
        f'Top-1 {a1:.4f} Top-5 {a5:.4f} Top-10 {a10:.4f} 1-way {w1:.4f} '
        f'best_epoch={best_epoch} val_top1={best_val:.4f} '
        f'feature={args.prefix}')
print(line, flush=True)
with open(result_path, 'a', encoding='utf-8') as f:
    f.write(line + '\n')
print(f'结果已追加: {result_path}', flush=True)
print(f'checkpoint: {ckpt_path}', flush=True)
print(f'总用时: {(time.time()-t0)/60:.1f} min', flush=True)
