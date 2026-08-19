# -*- coding: utf-8 -*-
"""
03_eval_100way.py — 用 best_retrieval.pth 做最终 100-way TEST 检索评估

协议：
1. 只使用 test_trials / test_imgs；
2. 每个 query = 1 张真图 + 99 张“不重复”的随机干扰图；
3. 按 brain embedding 与 CLIP embedding 的 cosine similarity 排序；
4. 3 个随机种子 × 2000 queries，输出 mean ± std。

注意：
- best_retrieval.pth 是根据 validation Top-1 选出的；
- test 集不参与模型选择。
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
N_QUERY = 2000
SEEDS = [0, 1, 2]

node_feat = np.load(os.path.join(HERE, 'node_features.npy'))
adj = np.load(os.path.join(HERE, 'adj.npy'))
trial_img = np.load(
    os.path.join(HERE, 'trial_img.npy')
).astype(np.int64)
test_trials = np.load(
    os.path.join(HERE, 'test_trials.npy')
).astype(np.int64)
clip = np.load(
    os.path.join(PROC, 'clip_embeddings.npy')
).astype(np.float32)

sys.path.insert(0, CODE_DIR)
from BrainGFM_Gprompt import BrainGFM


# 训练阶段已固定 moe_num_experts=1，因此最终评估也固定为 1。
class RetModel(nn.Module):
    def __init__(self, moe):
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
            moe_num_experts=moe,
        )

        self.projector = nn.Sequential(
            nn.Linear(HIDDEN, 256),
            nn.ReLU(),
            nn.Linear(256, CLIP_D),
        )

    def forward(self, x, a, valid):
        g = self.encoder(
            x,
            a,
            parc_type='schaefer',
            disease_type='none',
            valid_num_nodes=valid
        )
        return self.projector(g)


ckpt_path = os.path.join(HERE, 'best_retrieval.pth')
assert os.path.isfile(
    ckpt_path
), '找不到 %s，请先运行 02_train_retrieval.py' % ckpt_path

sd = torch.load(
    ckpt_path,
    map_location='cpu',
    weights_only=False
)

model = RetModel(1).to(DEVICE)
model.load_state_dict(sd, strict=True)
model.eval()

print('已加载:', ckpt_path)
print('DEVICE:', DEVICE)

test_imgs = np.unique(trial_img[test_trials])
clip_t = torch.from_numpy(clip).float()
adj_t = torch.from_numpy(adj).float().unsqueeze(0).to(DEVICE)

assert len(test_imgs) >= 100, len(test_imgs)


@torch.no_grad()
def embed(ts):
    e = np.zeros(
        (len(ts), CLIP_D),
        dtype=np.float32
    )

    for i0 in range(0, len(ts), 32):
        idx = ts[i0:i0 + 32]

        x = torch.from_numpy(
            node_feat[idx]
        ).float().to(DEVICE)

        a = adj_t.expand(
            len(idx), -1, -1
        )

        o = model(
            x,
            a,
            [100] * len(idx)
        )

        e[i0:i0 + 32] = F.normalize(
            o,
            dim=-1
        ).cpu().numpy()

    return e


def run(seed, nq):
    rng = np.random.default_rng(seed)

    nq = min(nq, len(test_trials))
    q = rng.choice(
        len(test_trials),
        size=nq,
        replace=False
    )

    ts = test_trials[q]
    embs = embed(ts)

    t1 = t5 = t10 = 0

    for i in range(len(ts)):
        cid = int(trial_img[ts[i]])

        # 不重复地抽 99 个 distractors
        pool = test_imgs[test_imgs != cid]

        distractors = rng.choice(
            pool,
            size=99,
            replace=False
        )

        cand = np.concatenate(
            [
                np.array([cid], dtype=np.int64),
                distractors
            ]
        )

        ce = F.normalize(
            clip_t[cand],
            dim=-1
        )

        qe = torch.from_numpy(
            embs[i]
        )

        sims = qe @ ce.T

        rank = int(
            torch.argsort(
                sims,
                descending=True
            ).tolist().index(0)
        ) + 1

        if rank <= 1:
            t1 += 1
        if rank <= 5:
            t5 += 1
        if rank <= 10:
            t10 += 1

    n = len(ts)

    return (
        t1 / n * 100,
        t5 / n * 100,
        t10 / n * 100
    )


print(
    '评估 100-way TEST, %d queries/seed, seeds=%s ...'
    % (N_QUERY, SEEDS)
)

res = [
    run(seed, N_QUERY)
    for seed in SEEDS
]

arr = np.array(res)

mean = arr.mean(axis=0)
std = arr.std(axis=0)

out_lines = []

out_lines.append(
    '100-way TEST 检索评估 '
    '(%d queries/seed, seeds=%s)'
    % (N_QUERY, SEEDS)
)

out_lines.append(
    '随机水平 ≈ Top-1 1.0%% / '
    'Top-5 5.0%% / Top-10 10.0%%'
)

for seed, r in zip(SEEDS, res):
    out_lines.append(
        '  seed %d: Top-1=%.2f  Top-5=%.2f  Top-10=%.2f'
        % (seed, r[0], r[1], r[2])
    )

out_lines.append(
    '  MEAN±STD: '
    'Top-1=%.2f±%.2f  '
    'Top-5=%.2f±%.2f  '
    'Top-10=%.2f±%.2f'
    % (
        mean[0], std[0],
        mean[1], std[1],
        mean[2], std[2]
    )
)

out_lines.append(
    '参考（不同 CLIP/数据协议，不能直接视为严格同条件）： '
    'Omni-fMRI 6.93/14.12/23.08 | '
    'NeuroSTORM 3.00/11.67/20.83 | '
    '随机 1/5/10'
)

out = '\n'.join(out_lines)

print('\n' + out)

with open(
    os.path.join(HERE, 'eval_results.txt'),
    'w'
) as f:
    f.write(out + '\n')

print('✅ 已保存 eval_results.txt')
