# -*- coding: utf-8 -*-
"""
train_nsd_classify_kmeans.py
============================
自定 NSD 分类任务（监督版，k-means 语义聚类类别）。

背景：
  想定义「脑信号 -> COCO 80 类」的零样本分类，但 processed/clip_embeddings.npy
  由 MindEye/NSD 流程里的某个 CLIP 生成，具体模型不明，我们生成的文本嵌入
  （OpenCLIP ViT-B-32）与它对不上，导致 80 类文本标签退化（person 只排第 48）。
  于是改走**纯图像侧、自包含**的类别定义：在 CLIP 图像嵌入上做 k-means，
  把 73k 张图聚成 K 个语义簇，作为「自定类别」。

任务定义：
  输入  = roi_ts（k-means 100 脑区响应，[30000,100]）；
  输出  = 该试次图片所属的 CLIP 语义簇（K 选 1）。
  CLIP 图像嵌入本身有语义结构（检索能到 13.6%），k-means 簇 = 自定语义类别。

模型：线性探针（或小 MLP），roi_ts(100) -> K 类。
划分：NSD 官方 split，train 3000 试次(1000图) / test 27000 试次(9000图)，图不重叠；
      再从 train 按图抽 200 图做 val 早停。

指标：Top-1 / Top-5；对照随机基线 1/K、5/K 与多数类基线。

用法：
  python train_nsd_classify_kmeans.py --K 50 --probe linear
"""
import argparse
import os
import sys
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--K', type=int, default=50)
parser.add_argument('--probe', choices=['linear', 'mlp'], default='linear')
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_val_imgs', type=int, default=200)
args = parser.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from config import PROCESSED_DIR

PROC = str(PROCESSED_DIR)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
K = args.K

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# ---------------- 载入 ----------------
print('载入数据 ...')
roi = np.load(os.path.join(PROC, 'roi_ts.npy')).astype(np.float32)       # [30000,100]
trial_img = np.load(os.path.join(HERE, 'trial_img.npy')).astype(np.int64)
train_tr = np.load(os.path.join(HERE, 'train_trials.npy')).astype(np.int64)
test_tr = np.load(os.path.join(HERE, 'test_trials.npy')).astype(np.int64)
clip = np.load(os.path.join(PROC, 'clip_embeddings.npy')).astype(np.float32)  # [73000,512]

# ---------------- k-means 自定类别 ----------------
lab_file = os.path.join(HERE, 'kmeans_cluster_labels_%d.npy' % K)
if os.path.exists(lab_file):
    img_cluster = np.load(lab_file).astype(np.int64)
    print('  复用已有 k-means 标签:', lab_file)
else:
    from sklearn.cluster import KMeans
    print('  k-means 聚类 %d 类（73000 x 512, 需 1-2 分钟）...' % K)
    km = KMeans(n_clusters=K, random_state=args.seed, n_init=10, max_iter=300)
    img_cluster = km.fit_predict(clip)
    np.save(lab_file, img_cluster)
    np.save(os.path.join(HERE, 'kmeans_cluster_centers_%d.npy' % K),
            km.cluster_centers_)
    print('  已保存:', lab_file)

labels = img_cluster[trial_img].astype(np.int64)     # [30000] 每试次所属簇
y_train = labels[train_tr]
y_test = labels[test_tr]

# 簇大小分布（train 里）
cnt = np.bincount(y_train, minlength=K)
sizes = np.sort(cnt)[::-1]
print('  train 簇大小: 最大 %d / 中位 %d / 最小 %d' % (sizes[0], int(np.median(cnt[cnt > 0])), int(cnt.min())))

# ---------------- 按图抽 val ----------------
tr_imgs = np.unique(trial_img[train_tr])
rng = np.random.default_rng(args.seed)
rng.shuffle(tr_imgs)
val_imgs = set(tr_imgs[:args.n_val_imgs].tolist())
fit_mask = ~np.isin(trial_img[train_tr], list(val_imgs))
fit_tr = train_tr[fit_mask]
val_tr = train_tr[~fit_mask]
print('  fit 试次 %d / val 试次 %d / test 试次 %d' % (len(fit_tr), len(val_tr), len(test_tr)))

# ---------------- 特征标准化 ----------------
mu = roi[fit_tr].mean(0, keepdims=True)
sd = roi[fit_tr].std(0, keepdims=True) + 1e-6
X_fit = (roi[fit_tr] - mu) / sd
X_val = (roi[val_tr] - mu) / sd
X_test = (roi[test_tr] - mu) / sd

# ---------------- 模型 ----------------
if args.probe == 'linear':
    probe = nn.Linear(100, K)
else:
    probe = nn.Sequential(nn.Linear(100, 256), nn.ReLU(), nn.Dropout(0.2),
                          nn.Linear(256, K))
probe = probe.to(DEVICE)
opt = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)
print('  探针:', args.probe, '| 参数数:', sum(p.numel() for p in probe.parameters()))

X_fit_t = torch.from_numpy(X_fit).to(DEVICE)
X_val_t = torch.from_numpy(X_val).to(DEVICE)
X_test_t = torch.from_numpy(X_test).to(DEVICE)
y_fit_t = torch.from_numpy(y_train[fit_mask]).to(DEVICE)
y_val_t = torch.from_numpy(y_train[~fit_mask]).to(DEVICE)
y_test_t = torch.from_numpy(y_test).to(DEVICE)


@torch.no_grad()
def acc(X, y):
    probe.eval()
    logits = probe(X)
    top1 = (logits.argmax(1) == y).float().mean().item()
    top5 = sum(int(y[i].item() in logits.topk(5, dim=1).indices[i].tolist())
               for i in range(len(y))) / len(y)
    return top1 * 100, top5 * 100


# ---------------- 训练 ----------------
best_val, best_state = -1.0, None
for ep in range(1, args.epochs + 1):
    probe.train()
    opt.zero_grad()
    loss = F.cross_entropy(probe(X_fit_t), y_fit_t)
    loss.backward()
    opt.step()
    if ep % 20 == 0 or ep == args.epochs:
        v1, v5 = acc(X_val_t, y_val_t)
        if v1 > best_val:
            best_val, best_state = v1, {k: v.clone() for k, v in probe.state_dict().items()}
        if ep % 100 == 0 or ep == args.epochs:
            print('  epoch %3d | loss=%.4f | val T1=%.2f T5=%.2f' % (ep, loss.item(), v1, v5))

probe.load_state_dict(best_state)
t1, t5 = acc(X_test_t, y_test_t)

# 多数类基线
maj = int(torch.bincount(y_fit_t).argmax())
maj_base = float((y_test_t == maj).float().mean() * 100)

# 每图多数投票
preds = probe(X_test_t).argmax(1).cpu().numpy()
img_test = trial_img[test_tr]
true_test = labels[test_tr]
per_img = {}
for im, p in zip(img_test, preds):
    per_img.setdefault(int(im), []).append(int(p))
vote_hit = 0
for im, ps in per_img.items():
    true_lab = int(true_test[np.where(img_test == im)[0][0]])
    if int(np.bincount(ps).argmax()) == true_lab:
        vote_hit += 1
vote_acc = vote_hit / len(per_img) * 100

lines = [
    '自定 NSD 分类（k-means 语义聚类，%s 探针）: roi_ts(100) -> %d 个 CLIP 语义簇'
    % (args.probe, K),
    'train %d 试次(%d 图) / val %d 试次 / test %d 试次(%d 图，图不重叠)'
    % (len(fit_tr), len(np.unique(trial_img[fit_tr])), len(val_tr),
       len(test_tr), len(per_img)),
    '',
    '随机基线（%d 类均匀）: Top-1 = %.2f%% | Top-5 = %.2f%%' % (K, 100.0 / K, 500.0 / K),
    '多数类基线(train 最频簇): Top-1 = %.2f%%' % maj_base,
    '',
    'TEST  Top-1 = %.2f%% | Top-5 = %.2f%%' % (t1, t5),
    'TEST  每图多数投票 Top-1 = %.2f%%' % vote_acc,
]
out = '\n'.join(lines)
print('\n' + out)
with open(os.path.join(HERE, 'classify_kmeans_%d_%s.txt' % (K, args.probe)), 'w') as f:
    f.write(out + '\n')
print('\n✅ 已保存 classify_kmeans_%d_%s.txt' % (K, args.probe))
