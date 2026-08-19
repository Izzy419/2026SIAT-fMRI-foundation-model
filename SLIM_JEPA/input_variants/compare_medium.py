# -*- coding: utf-8 -*-
"""
compare_medium.py  中量三方案公平对比
只用已下载的完整session(1-11跳过4)的试次; 三方案用【同一批试次】+【同一个检索头】对比。
输出: 三方案的 Top-1/5/10 + 1-way 诊断。
用法:  CUDA_VISIBLE_DEVICES=1 python compare_medium.py
"""
import numpy as np
import torch
import torch.nn as nn
import sys
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from config import DATA_DIR

DATA = str(DATA_DIR)
SESSIONS = list(range(40))   # session1-11 跳过不完整的4(0-based)

# ============ 1. 中量试次集合(只在已处理session里的试次) ============
meta = np.load(f'{DATA}/trial_meta.npy')
train_idx = np.load(f'{DATA}/train_idx.npy')
test_idx = np.load(f'{DATA}/test_idx.npy')
train_pos = np.where(np.isin(meta[train_idx, 0], SESSIONS))[0]   # 在train_idx中的位置
test_pos  = np.where(np.isin(meta[test_idx, 0], SESSIONS))[0]
print('中量试次: 训练', len(train_pos), ' 测试', len(test_pos))

# ============ 2. CLIP目标配对(trial_imgidx: 每个试次->图片->CLIP嵌入) ============
trial_imgidx = np.load(f'{DATA}/trial_imgidx.npy')
clip_emb = np.load(f'{DATA}/clip_emb.npy')
clip_uidx = np.load(f'{DATA}/clip_unique_idx.npy')
id2pos = {int(c): p for p, c in enumerate(clip_uidx)}
clip_tr_all = np.stack([clip_emb[id2pos[int(trial_imgidx[t])]] for t in train_idx])  # 全部训练试次
test_img_pos_all = np.array([id2pos[int(trial_imgidx[t])] for t in test_idx])        # 全部测试试次
clip_n = clip_emb / (np.linalg.norm(clip_emb, axis=1, keepdims=True) + 1e-9)         # 归一化
test_unique = np.unique(test_img_pos_all)                                            # 唯一测试图(干扰池)

# ============ 3. 检索头(三方案共用同一个, 保证公平) ============
class ClipProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.GELU(),
            nn.Linear(in_dim, in_dim), nn.GELU(),
            nn.Linear(in_dim, out_dim))
    def forward(self, x):
        return nn.functional.normalize(self.mlp(x), dim=-1)

def info_nce(q, k, temp=0.1):
    logits = q @ k.T / temp
    labels = torch.arange(q.size(0)).cuda()
    return nn.CrossEntropyLoss()(logits, labels)

def train_head(feats, clip_tr, epochs=150, lr=3e-4):
    model = ClipProjector(feats.shape[1], clip_tr.shape[1]).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    X = torch.from_numpy(feats).float()
    Y = torch.from_numpy(clip_tr).float()
    dl = DataLoader(TensorDataset(X, Y), batch_size=128, shuffle=True)
    total = epochs * len(dl); warm = 5 * len(dl)
    def lr_lambda(step):
        if step < warm: return step / max(1, warm)
        prog = (step - warm) / max(1, total - warm)
        return 0.5 * (1.0 + np.cos(np.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    for ep in range(epochs):
        tot = 0
        for xb, yb in dl:
            xb, yb = xb.cuda(), yb.cuda()
            q = model(xb); yb = nn.functional.normalize(yb, dim=-1)
            loss = info_nce(q, yb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
    return model

def eval_100way(model, feats, img_pos):
    model.eval()
    n = len(feats); top1 = top5 = top10 = hit1 = 0
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for i in range(n):
            q = model(torch.from_numpy(feats[i:i+1]).cuda()).cpu().numpy()[0]
            gt_pos = img_pos[i]; gt = clip_n[gt_pos]
            others = test_unique[test_unique != gt_pos]
            dist = rng.choice(others, size=99, replace=False)
            pool = np.concatenate([np.array([gt]), clip_n[dist]])
            scores = np.array([(q @ p) / (np.linalg.norm(q)*np.linalg.norm(p)+1e-9) for p in pool])
            rank = int(np.argsort(-scores)[0])
            top1 += (rank == 0); top5 += (rank < 5); top10 += (rank < 10)
            d1 = rng.choice(others, size=1, replace=False)[0]
            p2 = np.array([gt, clip_n[d1]])
            s2 = np.array([(q @ p2[0])/(np.linalg.norm(q)*np.linalg.norm(p2[0])+1e-9),
                           (q @ p2[1])/(np.linalg.norm(q)*np.linalg.norm(p2[1])+1e-9)])
            hit1 += int(np.argmax(s2) == 0)
    return top1/n, top5/n, top10/n, hit1/n

# ============ 4. 三方案对比 ============
schemes = {
    '方案1(β-repeat)':    (f'{DATA}/feats_train.npy',  f'{DATA}/feats_test.npy'),
    '方案2(raw-time)':    (f'{DATA}/feats2_train.npy', f'{DATA}/feats2_test.npy'),
    '方案3(segmented-β)': (f'{DATA}/feats3_train.npy', f'{DATA}/feats3_test.npy'),
}
print('=' * 66)
print(f"{'方案':<18}{'Top-1':>8}{'Top-5':>8}{'Top-10':>9}{'1-way':>8}")
print('=' * 66)
for name, (ftr_f, fte_f) in schemes.items():
    ftr = np.load(ftr_f)[train_pos]      # 只取中量训练试次
    fte = np.load(fte_f)[test_pos]       # 只取中量测试试次
    model = train_head(ftr, clip_tr_all[train_pos])
    t1, t5, t10, w1 = eval_100way(model, fte, test_img_pos_all[test_pos])
    print(f"{name:<18}{t1:>8.2%}{t5:>8.2%}{t10:>9.2%}{w1:>8.2%}")
print('=' * 66)
print('随机水平: Top-1 1% / Top-5 5% / Top-10 10% / 1-way 50%')
print('对照: Omni-fMRI 6.93% / 14.12% / 23.08%')
