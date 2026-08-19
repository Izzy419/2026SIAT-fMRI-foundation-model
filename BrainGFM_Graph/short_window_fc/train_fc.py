# -*- coding: utf-8 -*-
"""
train_fc.py
===========
k-means Pearson FC → BrainGFM 检索训练（第三种方案）。

自己实现。节点特征 = 每试次的 100×100 Pearson 功能连接矩阵（第 i 行 = 脑区 i
到所有脑区的连接模式，共 100 维）；邻接 = (FC > 0.3) 阈值，随试次变化。

和方案B的区别：
  方案B：节点特征 = roi_ts 激活（每节点 1 维标量），邻接固定；
  本脚本：节点特征 = FC 连接（每节点 100 维连接向量），邻接随试次变。
协议（multi-positive InfoNCE + 100-way 评估）与方案B一致，便于公平对比。

输入：processed/train_dataset_pearson_kmeans.npy / test_dataset_pearson_kmeans.npy
      每个样本 {'node_feat': [100,100] float32, 'label': 73k图索引 int}
输出：best_retrieval_fc.pth / training_log_fc.txt

用法：python train_fc.py
"""
import os
import sys
import time
import random
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
N = 100                          # 节点数（k-means 100 簇）

BATCH_IMAGES = 16
EPOCHS = 100
LR = 1e-4
WARMUP = 10
TEMP = 0.07
VAL_QUERY = 500
SEED = 0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---------------- 数据 ----------------
print('加载 FC 数据集 ...')
train_raw = np.load(os.path.join(PROC, 'train_dataset_pearson_kmeans.npy'), allow_pickle=True)
test_raw = np.load(os.path.join(PROC, 'test_dataset_pearson_kmeans.npy'), allow_pickle=True)

train_fc = np.stack([d['node_feat'] for d in train_raw]).astype(np.float32)   # (2309,100,100)
train_lab = np.array([int(d['label']) for d in train_raw], dtype=np.int64)
test_fc = np.stack([d['node_feat'] for d in test_raw]).astype(np.float32)     # (21195,100,100)
test_lab = np.array([int(d['label']) for d in test_raw], dtype=np.int64)

clip = np.load(os.path.join(PROC, 'clip_embeddings.npy')).astype(np.float32)

print('  train_fc:', train_fc.shape, ' train_lab:', train_lab.shape)
print('  test_fc :', test_fc.shape, ' test_lab :', test_lab.shape)
print('  clip:', clip.shape)
assert train_lab.max() < clip.shape[0] and test_lab.max() < clip.shape[0]

# ---------------- 训练集按 IMAGE 划分 fit/val ----------------
train_imgs = np.unique(train_lab)
rng = np.random.default_rng(SEED)
shuffled = train_imgs.copy()
rng.shuffle(shuffled)
n_val_imgs = max(1, int(0.2 * len(train_imgs)))
val_imgs = np.sort(shuffled[:n_val_imgs])
fit_imgs = np.sort(shuffled[n_val_imgs:])

fit_mask = np.isin(train_lab, fit_imgs)
val_mask = np.isin(train_lab, val_imgs)
fit_idx = np.where(fit_mask)[0]
val_idx = np.where(val_mask)[0]
print('  fit 图 %d / val 图 %d，fit 样本 %d / val 样本 %d'
      % (len(fit_imgs), len(val_imgs), len(fit_idx), len(val_idx)))

fit_groups = {}
for i in fit_idx:
    fit_groups.setdefault(int(train_lab[i]), []).append(int(i))

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
opt = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.5, 0.999))
print('参数总数: %d' % sum(p.numel() for p in model.parameters()))


def make_adj(fc_batch):
    """邻接 = (FC > 0.3)，随试次变化。"""
    return (fc_batch > 0.3).float()


def multi_positive_infonce(brain_emb, clip_emb, image_ids, temp):
    brain_emb = F.normalize(brain_emb, dim=-1)
    clip_emb = F.normalize(clip_emb, dim=-1)
    logits = brain_emb @ clip_emb.T / temp
    image_ids = image_ids.view(-1)
    pos_mask = image_ids[:, None].eq(image_ids[None, :])
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos_count = pos_mask.sum(dim=1).clamp_min(1)
    return -((log_prob * pos_mask.float()).sum(dim=1) / pos_count).mean()


@torch.no_grad()
def embed(fc_arr, idx):
    """把一批 FC 样本编码成归一化 embedding。"""
    e = np.zeros((len(idx), CLIP_D), dtype=np.float32)
    for i0 in range(0, len(idx), 32):
        ids = idx[i0:i0 + 32]
        x = torch.from_numpy(fc_arr[ids]).float().to(DEVICE)      # [b,100,100]
        a = make_adj(x)
        o = model(x, a, [N] * len(ids))
        e[i0:i0 + 32] = F.normalize(o, dim=-1).cpu().numpy()
    return e


@torch.no_grad()
def eval_100way(fc_arr, lab_arr, idx_pool, n_query, seed):
    """100-way 检索：1 真图 + 99 干扰图。"""
    model.eval()
    rng = np.random.default_rng(seed)
    n_query = min(n_query, len(idx_pool))
    q = rng.choice(idx_pool, size=n_query, replace=False)
    embs = embed(fc_arr, q)
    clip_t = torch.from_numpy(clip).float()
    cand_imgs = np.unique(lab_arr)
    t1 = t5 = t10 = 0

    for i in range(len(q)):
        cid = int(lab_arr[q[i]])
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


# ---------------- 训练 ----------------
logf = os.path.join(HERE, 'training_log_fc.txt')
with open(logf, 'w') as f:
    f.write('k-means Pearson FC -> BrainGFM, InfoNCE (N=%d)\n' % N)

best_val_top1 = -1.0
print('开始训练: %d epochs, N=%d, 节点特征=FC(100x100)' % (EPOCHS, N))

for ep in range(1, EPOCHS + 1):
    if ep <= WARMUP:
        lr = LR * ep / WARMUP
    else:
        p = (ep - WARMUP) / max(1, EPOCHS - WARMUP)
        lr = LR * 0.5 * (1 + np.cos(np.pi * p))
    for g in opt.param_groups:
        g['lr'] = lr

    model.train()
    imgs = list(fit_groups.keys())
    random.shuffle(imgs)
    tl, nb = 0.0, 0
    t0 = time.time()

    for b0 in range(0, len(imgs), BATCH_IMAGES):
        batch_imgs = imgs[b0:b0 + BATCH_IMAGES]
        idx_list, img_list = [], []
        for im in batch_imgs:
            for t in fit_groups[im]:
                idx_list.append(t)
                img_list.append(im)
        idx = np.asarray(idx_list, dtype=np.int64)
        img_ids = torch.from_numpy(np.asarray(img_list, dtype=np.int64)).to(DEVICE)

        x = torch.from_numpy(train_fc[idx]).float().to(DEVICE)
        a = make_adj(x)
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
        ep, EPOCHS, tl / max(nb, 1), lr, time.time() - t0)

    v1, v5, v10 = eval_100way(train_fc, train_lab, val_idx, VAL_QUERY, seed=1000 + ep)
    line += ' | VAL 100way T1/5/10: %.2f/%.2f/%.2f' % (v1, v5, v10)

    if v1 > best_val_top1:
        best_val_top1 = v1
        torch.save(model.state_dict(), os.path.join(HERE, 'best_retrieval_fc.pth'))
        line += '  <<< best (saved)'

    print(line)
    with open(logf, 'a') as f:
        f.write(line + '\n')

print('✅ 训练完成。最佳 VAL 100way Top-1 = %.2f%%' % best_val_top1)
print('   最终 TEST 评估: python eval_fc.py')
