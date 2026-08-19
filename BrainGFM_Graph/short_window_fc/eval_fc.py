# -*- coding: utf-8 -*-
"""
eval_fc.py
==========
用 best_retrieval_fc.pth 做最终 100-way TEST 检索评估（FC 方案）。

协议与方案B一致：3 个随机种子 × 2000 queries，100-way，输出 mean ± std。

用法：python eval_fc.py
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from config import MODEL_DIR, PROCESSED_DIR

PROC = str(PROCESSED_DIR)
CODE_DIR = str(MODEL_DIR)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
HIDDEN, CLIP_D = 256, 512
N = 100
N_QUERY = 2000
SEEDS = [0, 1, 2]

print('加载数据 ...')
train_raw = np.load(os.path.join(PROC, 'train_dataset_pearson_kmeans.npy'), allow_pickle=True)
test_raw = np.load(os.path.join(PROC, 'test_dataset_pearson_kmeans.npy'), allow_pickle=True)
test_fc = np.stack([d['node_feat'] for d in test_raw]).astype(np.float32)
test_lab = np.array([int(d['label']) for d in test_raw], dtype=np.int64)
clip = np.load(os.path.join(PROC, 'clip_embeddings.npy')).astype(np.float32)
print('  test_fc:', test_fc.shape, '  test_lab:', test_lab.shape)

sys.path.insert(0, CODE_DIR)
from BrainGFM_Gprompt import BrainGFM


class RetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = BrainGFM(
            hidden_dim=HIDDEN, ff_hidden_size=256, num_classes=2,
            num_self_att_layers=4, dropout=0.3, num_GNN_layers=4, nhead=8,
            max_feature_dim=512, rwse_steps=5, max_nodes=512, moe_num_experts=1)
        self.projector = nn.Sequential(
            nn.Linear(HIDDEN, 256), nn.ReLU(), nn.Linear(256, CLIP_D))

    def forward(self, x, a, valid):
        g = self.encoder(x, a, parc_type='schaefer',
                         disease_type='none', valid_num_nodes=valid)
        return self.projector(g)


ckpt_path = os.path.join(HERE, 'best_retrieval_fc.pth')
assert os.path.isfile(ckpt_path), '找不到 %s，先运行 train_fc.py' % ckpt_path
model = RetModel().to(DEVICE)
model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=False), strict=True)
model.eval()
print('已加载:', ckpt_path, '| DEVICE:', DEVICE)


@torch.no_grad()
def embed(ts):
    e = np.zeros((len(ts), CLIP_D), dtype=np.float32)
    for i0 in range(0, len(ts), 32):
        ids = ts[i0:i0 + 32]
        x = torch.from_numpy(test_fc[ids]).float().to(DEVICE)
        a = (x > 0.3).float()
        o = model(x, a, [N] * len(ids))
        e[i0:i0 + 32] = F.normalize(o, dim=-1).cpu().numpy()
    return e


def run(seed, nq):
    rng = np.random.default_rng(seed)
    nq = min(nq, len(test_lab))
    q = rng.choice(len(test_lab), size=nq, replace=False)
    embs = embed(q)
    clip_t = torch.from_numpy(clip).float()
    cand_imgs = np.unique(test_lab)
    t1 = t5 = t10 = 0

    for i in range(len(q)):
        cid = int(test_lab[q[i]])
        pool = cand_imgs[cand_imgs != cid]
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

    n = len(q)
    return t1 / n * 100, t5 / n * 100, t10 / n * 100


print('评估 100-way TEST, %d queries/seed, seeds=%s ...' % (N_QUERY, SEEDS))
res = [run(seed, N_QUERY) for seed in SEEDS]
arr = np.array(res)
mean, std = arr.mean(axis=0), arr.std(axis=0)

lines = [
    '100-way TEST 检索评估 (FC, N=%d, %d queries/seed, seeds=%s)' % (N, N_QUERY, SEEDS),
    '随机水平 ≈ Top-1 1.0%% / Top-5 5.0%% / Top-10 10.0%%',
]
for seed, r in zip(SEEDS, res):
    lines.append('  seed %d: Top-1=%.2f  Top-5=%.2f  Top-10=%.2f' % (seed, r[0], r[1], r[2]))
lines.append('  MEAN±STD: Top-1=%.2f±%.2f  Top-5=%.2f±%.2f  Top-10=%.2f±%.2f'
             % (mean[0], std[0], mean[1], std[1], mean[2], std[2]))
lines.append('  参考: 方案B(k-means roi_ts) ~13.6 | 随机 1/5/10')
out = '\n'.join(lines)
print('\n' + out)
with open(os.path.join(HERE, 'eval_results_fc.txt'), 'w') as f:
    f.write(out + '\n')
print('✅ 已保存 eval_results_fc.txt')
