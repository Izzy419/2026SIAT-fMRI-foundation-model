# -*- coding: utf-8 -*-
"""
eval_extra_metrics.py  多种评估方法(在已有 方案1 projector.pth 上)
- 100-way Top-1/5/10 + 1-way (原有)
- MRR: 真图排名的倒数平均 (随机≈0.01)
- 嵌入对齐: 预测脑嵌入 vs 真图CLIP嵌入 的皮尔逊相关 (衡量对齐质量, 与候选池无关)
- 排名分布: Top-k 各k的命中率 (k=1,2,3,5,10,20,50)
- 逐图可靠性: 同一张图在测试中出现多次(约3次), 其排名的一致性
输出: 结果追加 weekend_results.txt
用法:  CUDA_VISIBLE_DEVICES=1 python eval_extra_metrics.py
"""
import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
import numpy as np
import torch
import torch.nn as nn
from config import DATA_DIR

DATA = str(DATA_DIR)
feats = np.load(f'{DATA}/feats_test.npy')          # (27000,512) 方案1
test_idx = np.load(f'{DATA}/test_idx.npy')
trial_imgidx = np.load(f'{DATA}/trial_imgidx.npy')
clip_emb = np.load(f'{DATA}/clip_emb.npy')
clip_uidx = np.load(f'{DATA}/clip_unique_idx.npy')
id2pos = {int(c): p for p, c in enumerate(clip_uidx)}
test_img_pos = np.array([id2pos[int(trial_imgidx[t])] for t in test_idx])
test_unique = np.unique(test_img_pos)
clip_n = clip_emb / (np.linalg.norm(clip_emb, axis=1, keepdims=True) + 1e-9)

class ClipProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.GELU(),
            nn.Linear(in_dim, in_dim), nn.GELU(),
            nn.Linear(in_dim, out_dim))
    def forward(self, x):
        return nn.functional.normalize(self.mlp(x), dim=-1)

model = ClipProjector(512, 1280)
model.load_state_dict(torch.load(f'{DATA}/projector.pth', map_location='cuda'))
model.eval().cuda()

n = len(feats)
rng = np.random.default_rng(0)
ranks = np.zeros(n, dtype=int)      # 每个试次真图的排名(0-based)
q_all = np.zeros((n, 1280), dtype=np.float32)
gt_pos_all = np.zeros(n, dtype=int)
with torch.no_grad():
    for i in range(n):
        q = model(torch.from_numpy(feats[i:i+1]).cuda()).cpu().numpy()[0]
        q_all[i] = q
        gt_pos = test_img_pos[i]; gt_pos_all[i] = gt_pos
        gt = clip_n[gt_pos]
        others = test_unique[test_unique != gt_pos]
        dist = rng.choice(others, size=99, replace=False)
        pool = np.concatenate([np.array([gt]), clip_n[dist]])
        scores = pool @ q
        ranks[i] = int(np.argsort(-scores)[0])
        if (i+1) % 5000 == 0:
            print(f'  {i+1}/{n}', flush=True)

# ============ 汇总 ============
out = []
def add(name, val):
    out.append(f'  {name}: {val}')
    print(f'  {name}: {val}')

add('Top-1', f'{np.mean(ranks==0)*100:.2f}%')
add('Top-5', f'{np.mean(ranks<5)*100:.2f}%')
add('Top-10', f'{np.mean(ranks<10)*100:.2f}%')
# MRR
mrr = np.mean(1.0 / (ranks + 1))
add('MRR', f'{mrr:.4f} (随机≈0.01)')
# 排名分布
add('Top-k 分布', ' / '.join(f'{k}:{np.mean(ranks<k)*100:.1f}%' for k in [1,2,3,5,10,20,50]))
# 嵌入对齐: q 与 真图CLIP嵌入 的皮尔逊相关
gt_emb = clip_n[gt_pos_all]
qs = q_all
# 相关 = (q-mean)(gt-mean) / (std*std) 逐试次
mu_q = qs.mean(1, keepdims=True); mu_g = gt_emb.mean(1, keepdims=True)
cov = ((qs - mu_q) * (gt_emb - mu_g)).sum(1)
std_q = np.sqrt(((qs - mu_q)**2).sum(1)); std_g = np.sqrt(((gt_emb - mu_g)**2).sum(1))
corr = cov / (std_q * std_g + 1e-9)
add('嵌入对齐(皮尔逊r, q vs 真图)', f'{np.mean(corr):.4f} ± {np.std(corr):.4f} (随机≈0)')
# 逐图可靠性: 同一图多次出现, 其排名的一致性
unique_img, counts = np.unique(gt_pos_all, return_counts=True)
rep = unique_img[counts >= 2]
rank_by_img = {u: ranks[gt_pos_all == u] for u in rep}
within_std = np.array([r.std() for r in rank_by_img.values()])
mean_rank_by_img = np.array([r.mean() for r in rank_by_img.values()])
add('重复图数量(≥2次)', f'{len(rep)}')
add('逐图排名一致性(图内排名std)', f'{np.mean(within_std):.2f} (越小越稳定)')
add('逐图平均排名', f'{np.mean(mean_rank_by_img):.1f} (随机≈50)')

with open(f'{DATA}/weekend_results.txt', 'a', encoding='utf-8') as f:
    f.write('[多种评估方法 方案1]\n')
    for l in out:
        f.write(l + '\n')
print('多种评估方法 完成', flush=True)
