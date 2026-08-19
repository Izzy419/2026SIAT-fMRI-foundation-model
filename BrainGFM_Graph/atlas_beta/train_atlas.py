# -*- coding: utf-8 -*-
"""
train_atlas.py
==============
BrainGFM + NSD 100-way 检索训练（图谱方案）。

自己实现的训练脚本。BrainGFM 是固定库（从 code/BrainGFM 导入），我们只写
数据读取 + 训练循环。节点数 N 自动从数据读出（AAL=35 / Schaefer=26），
不再写死 100。

关键设计（对齐方案B，保证公平对比）：
  - multi-positive InfoNCE：同一图片的 3 个 trial 互为 positive
  - 训练集内部按 IMAGE 划分 train/val：800/200 图
  - 100-way validation 选 best（只看 val，不看 test）
  - 从零训练（moe_num_experts=1）

输入：build_dataset_atlas.py 的输出
输出：best_retrieval_<atlas>.pth / training_log_<atlas>.txt / split_info_<atlas>.npz

用法：python train_atlas.py --atlas aal
"""
import argparse
import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--atlas', choices=['aal', 'schaefer'], default='aal')
parser.add_argument('--epochs', type=int, default=150)
args = parser.parse_args()
ATLAS = args.atlas

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from config import MODEL_DIR, PROCESSED_DIR

PROC = str(PROCESSED_DIR)
CODE_DIR = str(MODEL_DIR)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
HIDDEN, CLIP_D = 256, 512

# 超参数（和方案B一致）
BATCH_IMAGES = 16
LR = 1e-4
WARMUP = 10
TEMP = 0.07
VAL_IMAGES = 200
TRAIN_IMAGES = 800
VAL_QUERY = 500
SEED = 0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---------------- 数据 ----------------
print('加载数据 ...')
node_feat = np.load(os.path.join(HERE, 'node_features_%s.npy' % ATLAS))  # [30000,N,1]
adj = np.load(os.path.join(HERE, 'adj_%s.npy' % ATLAS))                  # [N,N]
trial_img = np.load(os.path.join(HERE, 'trial_img.npy')).astype(np.int64)
train_trials = np.load(os.path.join(HERE, 'train_trials.npy')).astype(np.int64)
test_trials = np.load(os.path.join(HERE, 'test_trials.npy')).astype(np.int64)
clip = np.load(os.path.join(PROC, 'clip_embeddings.npy')).astype(np.float32)

N = node_feat.shape[1]                       # 节点数：AAL=35 / Schaefer=26
print('  node_feat:', node_feat.shape, '-> 节点数 N =', N)
print('  adj:', adj.shape)
print('  train_trials:', len(train_trials), ' test_trials:', len(test_trials))
assert len(train_trials) == 3000 and len(test_trials) == 27000
assert N <= 512, '节点数超过模型 max_nodes=512'

# ---------------- 按 IMAGE 划分 train / val ----------------
train_all_imgs = np.unique(trial_img[train_trials])
test_imgs = np.unique(trial_img[test_trials])
assert len(train_all_imgs) == 1000 and len(test_imgs) == 9000

rng_split = np.random.default_rng(SEED)
shuffled_imgs = train_all_imgs.copy()
rng_split.shuffle(shuffled_imgs)
val_imgs = np.sort(shuffled_imgs[:VAL_IMAGES])
fit_imgs = np.sort(shuffled_imgs[VAL_IMAGES:])

fit_mask = np.isin(trial_img[train_trials], fit_imgs)
val_mask = np.isin(trial_img[train_trials], val_imgs)
fit_trials = train_trials[fit_mask]
val_trials = train_trials[val_mask]

print('TRAIN images/trials:', len(fit_imgs), len(fit_trials))
print('VAL   images/trials:', len(val_imgs), len(val_trials))
print('TEST  images/trials:', len(test_imgs), len(test_trials))

np.savez(
    os.path.join(HERE, 'split_info_%s.npz' % ATLAS),
    fit_imgs=fit_imgs, val_imgs=val_imgs, test_imgs=test_imgs,
    fit_trials=fit_trials, val_trials=val_trials,
)

# 每张图 -> 它的 3 个 trial
fit_groups = {}
for t in fit_trials.tolist():
    fit_groups.setdefault(int(trial_img[t]), []).append(int(t))
val_groups = {}
for t in val_trials.tolist():
    val_groups.setdefault(int(trial_img[t]), []).append(int(t))
for im, ts in fit_groups.items():
    assert len(ts) == 3, '图片 %d 的 trial 数不是 3: %d' % (im, len(ts))

# ---------------- 模型 ----------------
sys.path.insert(0, CODE_DIR)
from BrainGFM_Gprompt import BrainGFM


class RetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = BrainGFM(
            hidden_dim=HIDDEN,
            ff_hidden_size=256,
            num_classes=2,
            num_self_att_layers=4,
            dropout=0.3,
            num_GNN_layers=4,
            nhead=8,
            max_feature_dim=512,
            rwse_steps=5,
            max_nodes=512,
            moe_num_experts=1,
        )
        self.projector = nn.Sequential(
            nn.Linear(HIDDEN, 256),
            nn.ReLU(),
            nn.Linear(256, CLIP_D),
        )

    def forward(self, x, a, valid):
        g = self.encoder(x, a, parc_type='schaefer',
                         disease_type='none', valid_num_nodes=valid)
        return self.projector(g)


model = RetModel().to(DEVICE)
tot = sum(p.numel() for p in model.parameters())
print('参数总数: %d' % tot)

opt = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.5, 0.999))
adj_t = torch.from_numpy(adj).float().unsqueeze(0).to(DEVICE)


# ---------------- multi-positive InfoNCE ----------------
def multi_positive_infonce(brain_emb, clip_emb, image_ids, temp):
    brain_emb = F.normalize(brain_emb, dim=-1)
    clip_emb = F.normalize(clip_emb, dim=-1)
    logits = brain_emb @ clip_emb.T / temp
    image_ids = image_ids.view(-1)
    pos_mask = image_ids[:, None].eq(image_ids[None, :])
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos_count = pos_mask.sum(dim=1).clamp_min(1)
    return -((log_prob * pos_mask.float()).sum(dim=1) / pos_count).mean()


def make_batches():
    imgs = list(fit_groups.keys())
    random.shuffle(imgs)
    batches, cur = [], []
    for im in imgs:
        cur.append(im)
        if len(cur) == BATCH_IMAGES:
            batches.append(cur)
            cur = []
    if cur:
        batches.append(cur)
    return batches


# ---------------- 100-way validation ----------------
@torch.no_grad()
def eval_100way(model, trials, candidate_imgs, n_query, seed):
    model.eval()
    rng = np.random.default_rng(seed)
    n_query = min(n_query, len(trials))
    qidx = rng.choice(len(trials), size=n_query, replace=False)
    qtrials = trials[qidx]
    embs = np.zeros((len(qtrials), CLIP_D), dtype=np.float32)

    for i0 in range(0, len(qtrials), 32):
        idx = qtrials[i0:i0 + 32]
        x = torch.from_numpy(node_feat[idx]).float().to(DEVICE)
        a = adj_t.expand(len(idx), -1, -1)
        e = model(x, a, [N] * len(idx))
        embs[i0:i0 + 32] = F.normalize(e, dim=-1).cpu().numpy()

    clip_t = torch.from_numpy(clip).float()
    candidate_imgs = np.asarray(candidate_imgs, dtype=np.int64)
    t1 = t5 = t10 = 0

    for i in range(len(qtrials)):
        cid = int(trial_img[qtrials[i]])
        pool = candidate_imgs[candidate_imgs != cid]
        distractors = rng.choice(pool, size=99, replace=False)
        cand = np.concatenate([np.array([cid], dtype=np.int64), distractors])
        ce = F.normalize(clip_t[cand], dim=-1)
        qe = torch.from_numpy(embs[i])
        sims = qe @ ce.T
        rank = int(torch.argsort(sims, descending=True).tolist().index(0)) + 1
        if rank <= 1:
            t1 += 1
        if rank <= 5:
            t5 += 1
        if rank <= 10:
            t10 += 1

    n = len(qtrials)
    return t1 / n * 100, t5 / n * 100, t10 / n * 100


# ---------------- 训练 ----------------
logf = os.path.join(HERE, 'training_log_%s.txt' % ATLAS)
with open(logf, 'w') as f:
    f.write('Multi-positive InfoNCE (atlas=%s, N=%d)\n' % (ATLAS, N))

best_val_top1 = -1.0
print('开始训练: %d epochs, N=%d 节点, lr=%.0e, temp=%.2f' % (args.epochs, N, LR, TEMP))

for ep in range(1, args.epochs + 1):
    if ep <= WARMUP:
        lr = LR * ep / WARMUP
    else:
        p = (ep - WARMUP) / max(1, args.epochs - WARMUP)
        lr = LR * 0.5 * (1 + np.cos(np.pi * p))
    for g in opt.param_groups:
        g['lr'] = lr

    model.train()
    tl, nb = 0.0, 0
    t0 = time.time()

    for batch_imgs in make_batches():
        idx_list, img_list = [], []
        for im in batch_imgs:
            for t in fit_groups[im]:
                idx_list.append(t)
                img_list.append(im)
        idx = np.asarray(idx_list, dtype=np.int64)
        img_ids = torch.from_numpy(np.asarray(img_list, dtype=np.int64)).to(DEVICE)

        x = torch.from_numpy(node_feat[idx]).float().to(DEVICE)
        a = adj_t.expand(len(idx), -1, -1)
        clip_tgt = torch.from_numpy(clip[np.asarray(img_list, dtype=np.int64)]).float().to(DEVICE)

        e = model(x, a, [N] * len(idx))
        loss = multi_positive_infonce(e, clip_tgt, img_ids, TEMP)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        tl += loss.item()
        nb += 1

    line = 'Epoch %3d/%d | loss=%.4f | lr=%.1e | %.1fs' % (
        ep, args.epochs, tl / max(nb, 1), lr, time.time() - t0)

    v1, v5, v10 = eval_100way(model, val_trials, val_imgs, VAL_QUERY, seed=1000 + ep)
    line += ' | VAL 100way T1/5/10: %.2f/%.2f/%.2f' % (v1, v5, v10)

    if v1 > best_val_top1:
        best_val_top1 = v1
        torch.save(model.state_dict(), os.path.join(HERE, 'best_retrieval_%s.pth' % ATLAS))
        line += '  <<< best (saved)'

    print(line)
    with open(logf, 'a') as f:
        f.write(line + '\n')

print('✅ 训练完成。最佳 VAL 100-way Top-1 = %.2f%%' % best_val_top1)
print('   最终 TEST 评估: python eval_atlas.py --atlas %s' % ATLAS)
