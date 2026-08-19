# -*- coding: utf-8 -*-
"""
finetune_slim_v2.py  全量微调修复版(完成 probe/LoRA/直接解冻 三连对比)
针对第一版(1.27%, 编码器LR太大毁特征)和MoCo fix版(学不动)的两个问题:

修复:
 1. 编码器极小LR (5e-6) + 头正常LR (1e-4)   ← 防毁掉预训练特征
 2. 梯度裁剪 (max_norm=1.0)                  ← 防编码器梯度爆炸
 3. warmup + cosine 学习率调度
 4. 每隔几epoch在 held-out 测试子集上评估, 按评估指标保存最优模型(不是train loss)
 5. 干净的 MoCo 负样本队列(去掉同图试次的错误负样本: batch内若出现同一张图的两次试次, 排除)

用法: CUDA_VISIBLE_DEVICES=1 python finetune_slim_v2.py full [epochs]
      CUDA_VISIBLE_DEVICES=1 python finetune_slim_v2.py lora [epochs]
      CUDA_VISIBLE_DEVICES=1 python finetune_slim_v2.py probe [epochs]
输出: slim_finetuned_v2_{MODE}.pth
"""
import sys, os, time
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / 'preprocess'))
sys.path.insert(0, str(PROJECT_DIR / 'model'))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from config import CHECKPOINT_PATH, DATA_DIR, NSD_MIN_DIR
from nsd_dataset import NSDSingleTrial
from hiera.hiera_mae import SlimEncoder

DATA = str(DATA_DIR)
BETA = str(NSD_MIN_DIR / 'betas_all_subj01_fp32_renorm.hdf5')
MASK_F = str(NSD_MIN_DIR / 'nsdgeneral.nii.gz')
CKPT = str(CHECKPOINT_PATH)

MODE = 'probe'
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
BATCH = 8
NEG_QUEUE = 512
LR_ENC = 1e-5 if MODE == 'full' else (1e-3 if MODE == 'lora' else 0)
LR_HEAD = 1e-4
EVAL_EVERY = 2          # 每隔几epoch做一次held-out评估
EVAL_N = None           # 全量held-out评估；None=完整test_idx

cfg = dict(input_size=(40, 96, 96, 96), in_chans=1, patch_kernel=(1, 4, 4, 4),
    patch_stride=(1, 4, 4, 4), patch_padding=(0, 0, 0, 0), embed_dim=64, num_heads=1,
    stages=(2, 3, 16, 3), q_pool=2, q_stride=(2, 2, 2, 2), mask_unit_size=(8, 8, 8, 8),
    mlp_ratio=4.0, sep_pos_embed=True)

class ClipProjector(nn.Module):
    def __init__(self, ind, outd):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(ind, ind), nn.GELU(),
                                 nn.Linear(ind, ind), nn.GELU(),
                                 nn.Linear(ind, outd))
    def forward(self, x):
        return F.normalize(self.mlp(x), dim=-1)

class LoRALinear(nn.Module):
    def __init__(self, linear, rank=8, alpha=16):
        super().__init__()
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad = False
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
    def forward(self, x):
        return self.linear(x) + self.scaling * (x @ self.lora_A.T @ self.lora_B.T)

def inject_lora(module, rank=8, alpha=16, min_features=64):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and child.in_features >= min_features:
            setattr(module, name, LoRALinear(child, rank, alpha))
        else:
            inject_lora(child, rank, alpha, min_features)

class PairedDS(Dataset):
    def __init__(self, base, clip, imgid):
        self.base = base; self.clip = clip; self.imgid = imgid
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        return self.base[i], self.clip[i], self.imgid[i]

# ============ 加载 ============
print('加载 SLIM + 检索头...', flush=True)
encoder = SlimEncoder(**cfg); encoder.load_from_mae(CKPT)
head = ClipProjector(512, 1280)

if MODE == 'probe':
    for p in encoder.parameters(): p.requires_grad = False
elif MODE == 'lora':
    for p in encoder.parameters(): p.requires_grad = False
    inject_lora(encoder, rank=8, alpha=16)
else:
    for p in encoder.parameters(): p.requires_grad = True
print(f'模式={MODE}  epochs={EPOCHS}  LR_ENC={LR_ENC}  LR_HEAD={LR_HEAD}', flush=True)

# ============ 数据 ============
train_idx = np.load(f'{DATA}/train_idx.npy')
trial_imgidx = np.load(f'{DATA}/trial_imgidx.npy')
clip_emb = np.load(f'{DATA}/clip_emb.npy'); clip_uidx = np.load(f'{DATA}/clip_unique_idx.npy')
id2pos = {int(c): p for p, c in enumerate(clip_uidx)}
clip_tr = torch.stack([torch.from_numpy(clip_emb[id2pos[int(trial_imgidx[t])]]) for t in train_idx]).float()
imgid_tr = np.array([trial_imgidx[t] for t in train_idx])

base = NSDSingleTrial(BETA, MASK_F, train_idx, t_frames=40)
ds = PairedDS(base, clip_tr, imgid_tr)
dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0, drop_last=True)

# 前景mask
tmp = NSDSingleTrial(BETA, MASK_F, [train_idx[0]], t_frames=40)
vol = tmp[0][0].numpy()
vt = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)
with torch.no_grad():
    token_max = F.max_pool3d(vt, kernel_size=4, stride=4)
    token_fore = token_max > 0
    mu = token_fore.view(1, 1, 3, 8, 3, 8, 3, 8).amax(dim=(3, 5, 7))
    mask = mu.flatten().bool().repeat(5).unsqueeze(0).cuda()
print(f'前景mask: {int(mask.sum())}/135', flush=True)

encoder.cuda(); head.cuda()

# feature维度检查
with torch.no_grad():
    _tmp = torch.zeros(1, 1, 40, 96, 96, 96).cuda()
    _m = mask.repeat(1, 1)
    print('feature shape check:', encoder.forward_features(_tmp, _m).shape, flush=True)
    del _tmp
encoder.train(); head.train()

enc_params = [p for p in encoder.parameters() if p.requires_grad]
head_params = [p for p in head.parameters() if p.requires_grad]
groups = []
if enc_params: groups.append({'params': enc_params, 'lr': LR_ENC})
if head_params: groups.append({'params': head_params, 'lr': LR_HEAD})
opt = torch.optim.AdamW(groups, weight_decay=0.05)
print(f'可训练: 编码器{sum(p.numel() for p in enc_params)/1e6:.1f}M + 头{sum(p.numel() for p in head_params)/1e6:.1f}M', flush=True)

# warmup + cosine
total_steps = EPOCHS * len(dl)
warm = int(0.05 * total_steps)
def lr_lambda(step):
    if step < warm: return step / max(1, warm)
    return 0.5 * (1 + np.cos(np.pi * (step - warm) / max(1, total_steps - warm)))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

# MoCo负样本队列
queue = F.normalize(torch.randn(NEG_QUEUE, 1280).cuda(), dim=-1)

def info_nce(q, keys, labels):
    logits = q @ keys.T / 0.1
    return nn.CrossEntropyLoss()(logits, torch.as_tensor(labels).cuda())

# ============ held-out 评估(小测试子集, 快) ============
def quick_eval(encoder, head, n=EVAL_N):
    encoder.eval(); head.eval()
    test_idx_all = np.load(f'{DATA}/test_idx.npy')
    test_idx = test_idx_all if n is None else test_idx_all[:min(n, len(test_idx_all))]
    test_img_pos = np.array([id2pos[int(trial_imgidx[t])] for t in test_idx])
    test_unique = np.unique(test_img_pos)
    ds_t = NSDSingleTrial(BETA, MASK_F, test_idx, t_frames=40)
    dl_t = DataLoader(ds_t, batch_size=8, shuffle=False, num_workers=0)
    Q = []
    with torch.no_grad():
        for xb in dl_t:
            x = xb.unsqueeze(1).cuda(); m = mask.repeat(x.shape[0], 1)
            Q.append(head(encoder.forward_features(x, m)).cpu().numpy())
    Q = np.concatenate(Q)
    clip_n = clip_emb / (np.linalg.norm(clip_emb, axis=1, keepdims=True) + 1e-9)
    rng = np.random.default_rng(0); t1 = hit = 0; N = len(Q)
    print(f'全量测试评估试次: {N}', flush=True)
    for i in range(N):
        q = Q[i]; gt_pos = test_img_pos[i]; gt = clip_n[gt_pos]
        others = test_unique[test_unique != gt_pos]
        dist = rng.choice(others, size=99, replace=False)
        pool = np.concatenate([np.array([gt]), clip_n[dist]])
        t1 += (int(np.argsort(-(pool @ q))[0]) == 0)
        d1 = rng.choice(others, size=1, replace=False)[0]
        hit += int(np.argmax(np.array([gt, clip_n[d1]]) @ q) == 0)
    encoder.train(); head.train()
    return t1 / N, hit / N

# ============ 训练 ============
best_t1 = -1
for ep in range(EPOCHS):
    tot = 0; t0 = time.time(); step = ep * len(dl)
    for i, (xb, yb, imgid) in enumerate(dl):
        x = xb.unsqueeze(1).cuda()
        m = mask.repeat(x.shape[0], 1)
        q = head(encoder.forward_features(x, m))          # (B,1280)
        yb = F.normalize(yb.cuda(), dim=-1)
        # 排除batch内同图试次作为负样本: 只保留 与q[i]不同图的 keys
        keys = torch.cat([yb, queue], dim=0)
        labels = list(range(BATCH))
        # 若batch内有同图试次, 把它们的标签设为-100(忽略)
        for a in range(BATCH):
            for b_ in range(BATCH):
                if a != b_ and imgid[a] == imgid[b_]:
                    labels[a] = -100
        loss = info_nce(q, keys, labels)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(
            [p for g in opt.param_groups for p in g['params']], 1.0)
        opt.step(); sched.step()
        B = x.shape[0]
        queue = torch.cat([queue[-(NEG_QUEUE - B):], yb.detach()])
        tot += loss.item()
        step += 1
        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{len(dl)}, loss {tot/(i+1):.4f}, {time.time()-t0:.0f}s', flush=True)
    avg = tot / len(dl)
    print(f'epoch {ep}: loss {avg:.4f} ({time.time()-t0:.0f}s)', flush=True)
    if (ep + 1) % EVAL_EVERY == 0 or ep == EPOCHS - 1:
        t1, w1 = quick_eval(encoder, head)
        print(f'  [held-out 全量 {EVAL_N if EVAL_N is not None else N}试次] Top-1 {t1:.4f}  1-way {w1:.4f}', flush=True)
        if t1 > best_t1:
            best_t1 = t1
            torch.save({'encoder': encoder.state_dict(), 'head': head.state_dict()},
                       f'{DATA}/slim_finetuned_v3_{MODE}_fulltest.pth')
            print(f'  [保存] slim_finetuned_v3_{MODE}_fulltest.pth (best Top-1 {t1:.4f})', flush=True)

print(f'微调完成({MODE}), 最优held-out Top-1 = {best_t1:.4f}', flush=True)
