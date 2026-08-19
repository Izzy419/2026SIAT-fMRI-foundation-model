# -*- coding: utf-8 -*-
"""
eval_multiseed.py  多种子稳定性: 方案1检索头用不同随机种子训练5次, 报 mean±std
对照 Omni 论文"三次运行平均±std"的报告方式。
输出: 结果追加到 weekend_results.txt
用法:  CUDA_VISIBLE_DEVICES=1 python eval_multiseed.py
"""
import sys, time
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from config import DATA_DIR

DATA = str(DATA_DIR)
EPOCHS, LR, BATCH, WARMUP = 150, 3e-4, 128, 5
TEMP = 0.1
SEEDS = [0, 1, 2, 3, 4]

feats = np.load(f'{DATA}/feats_train.npy')
feats_test = np.load(f'{DATA}/feats_test.npy')
train_idx = np.load(f'{DATA}/train_idx.npy')
test_idx = np.load(f'{DATA}/test_idx.npy')
trial_imgidx = np.load(f'{DATA}/trial_imgidx.npy')
clip_emb = np.load(f'{DATA}/clip_emb.npy')
clip_uidx = np.load(f'{DATA}/clip_unique_idx.npy')
id2pos = {int(c): p for p, c in enumerate(clip_uidx)}
clip_tr = np.stack([clip_emb[id2pos[int(trial_imgidx[t])]] for t in train_idx])
clip_tr_n = clip_tr / (np.linalg.norm(clip_tr, axis=1, keepdims=True) + 1e-9)
test_img_pos = np.array([id2pos[int(trial_imgidx[t])] for t in test_idx])
test_unique = np.unique(test_img_pos)
clip_n = clip_emb / (np.linalg.norm(clip_emb, axis=1, keepdims=True) + 1e-9)
print('特征:', feats.shape, flush=True)

class ClipProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.GELU(),
            nn.Linear(in_dim, in_dim), nn.GELU(),
            nn.Linear(in_dim, out_dim))
    def forward(self, x):
        return nn.functional.normalize(self.mlp(x), dim=-1)

def info_nce(q, k, temp=TEMP):
    return nn.CrossEntropyLoss()(q @ k.T / temp, torch.arange(q.size(0)).cuda())

def train_head(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ClipProjector(feats.shape[1], clip_tr.shape[1]).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    X = torch.from_numpy(feats).float(); Y = torch.from_numpy(clip_tr_n).float()
    dl = DataLoader(TensorDataset(X, Y), batch_size=BATCH, shuffle=True)
    total = EPOCHS * len(dl); warm = WARMUP * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s/max(1, warm) if s < warm
        else 0.5*(1 + np.cos(np.pi*(s-warm)/max(1, total-warm))))
    best = float('inf')
    for ep in range(EPOCHS):
        tot = 0.0
        for xb, yb in dl:
            xb, yb = xb.cuda(), yb.cuda()
            q = model(xb); yb = nn.functional.normalize(yb, dim=-1)
            loss = info_nce(q, yb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
        if tot/len(dl) < best:
            best = tot/len(dl)
            torch.save(model.state_dict(), f'{DATA}/_multiseed_{seed}.pth')
    return model

def eval_100way(model, feats, img_pos, n=None, rng_seed=0):
    model.eval()
    n = n or len(feats)
    t1 = t5 = t10 = hit = 0
    rng = np.random.default_rng(rng_seed)
    with torch.no_grad():
        for i in range(n):
            q = model(torch.from_numpy(feats[i:i+1]).cuda()).cpu().numpy()[0]
            gt_pos = img_pos[i]; gt = clip_n[gt_pos]
            others = test_unique[test_unique != gt_pos]
            dist = rng.choice(others, size=99, replace=False)
            pool = np.concatenate([np.array([gt]), clip_n[dist]])
            scores = pool @ q
            rank = int(np.argsort(-scores)[0])
            t1 += (rank==0); t5 += (rank<5); t10 += (rank<10)
            d1 = rng.choice(others, size=1, replace=False)[0]
            p2 = np.array([gt, clip_n[d1]])
            hit += int(np.argmax(p2 @ q) == 0)
    return t1/n, t5/n, t10/n, hit/n

results = []
for seed in SEEDS:
    print(f'训练 seed {seed}...', flush=True)
    m = train_head(seed)
    a1, a5, a10, w1 = eval_100way(m, feats_test, test_img_pos)
    print(f'  seed {seed}: Top-1 {a1:.4f} Top-5 {a5:.4f} Top-10 {a10:.4f} 1-way {w1:.4f}', flush=True)
    results.append([a1, a5, a10, w1])

r = np.array(results)
means, stds = r.mean(0), r.std(0)
line = (f'[多种子] 方案1 Top-1 {means[0]*100:.2f}±{stds[0]*100:.2f}% '
        f'Top-5 {means[1]*100:.2f}±{stds[1]*100:.2f}% '
        f'Top-10 {means[2]*100:.2f}±{stds[2]*100:.2f}% '
        f'1-way {means[3]*100:.2f}±{stds[3]*100:.2f}% '
        f'(5个种子 {SEEDS})')
print(line, flush=True)
with open(f'{DATA}/weekend_results.txt', 'a', encoding='utf-8') as f:
    f.write(line + '\n')
print('多种子完成', flush=True)
