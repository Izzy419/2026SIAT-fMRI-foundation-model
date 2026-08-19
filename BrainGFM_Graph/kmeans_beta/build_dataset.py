# -*- coding: utf-8 -*-
"""
01_build_dataset.py — 重建 NSD 检索数据集（方案B）

设计（为什么这样改）：
  luokexin 原管线里 node_feat = 每试次 ROI 响应的"外积"fc = x·xᵀ，是 rank-1 退化矩阵；
  且 8/6 那版满秩"Pearson FC"实测只有约 27% 试次特异性（73% 是跨试次共享结构），
  刺激信号被稀释，导致检索只能到随机水平。
  方案B（复现指导 5.1 / 你们 build_graphs.py 的思路）：
      节点特征 = 每试次 ROI 响应向量 (100 维, z-score, 用训练试次统计防泄漏)  —— 保留 100% 试次信号
      邻接矩阵 = 跨训练试次的 ROI 相关（固定，不随试次变）, 阈值 >0.3（与 BrainGFM 仓库一致, 无 abs）
      标签     = COCO_73k_subj_indices['subj01']（73k 图索引）

输入（只读父目录，勿改）：
  ../processed/roi_ts.npy                    [30000,100] 每试次 k-means 100 区响应
  ../processed/subj01_train_trial_indices.npy [3000]
  ../processed/subj01_test_trial_indices.npy  [27000]
  ../COCO_73k_subj_indices.hdf5              ['subj01'][30000] 试次→73k图索引

输出（写本脚本所在目录 besides/）：
  node_features.npy [30000,100,1]  float32  z-score 后节点特征
  adj.npy           [100,100]      float32  固定邻接(0/1, 对角=1)
  trial_img.npy     [30000]        int64    每试次图片(73k索引)
  train_trials.npy  [3000]         int64
  test_trials.npy   [27000]        int64
  train_stats.npz                       z-score 统计量(复现用)

用法：<env>/bin/python 01_build_dataset.py
"""
import os
import sys
from pathlib import Path
import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from config import DATA_DIR, PROCESSED_DIR

DATA = str(DATA_DIR)
PROC = str(PROCESSED_DIR)
OUT = HERE

N_REGIONS = 100
assert os.path.isdir(PROC), '找不到 processed 目录: %s' % PROC

# ---------- 1) 每试次 ROI 响应 ----------
print('[1] 读取 per-trial ROI 响应 ...')
roi = np.load(os.path.join(PROC, 'roi_ts.npy'))          # [30000,100]
assert roi.shape == (30000, N_REGIONS), roi.shape
print('    roi_ts:', roi.shape, roi.dtype)

# ---------- 2) 划分（已修正的正确划分, _correct 版） ----------
print('[2] 读取训练/测试试次索引 ...')
tr = np.load(os.path.join(PROC, 'subj01_train_trial_indices.npy')).astype(np.int64)
te = np.load(os.path.join(PROC, 'subj01_test_trial_indices.npy')).astype(np.int64)
assert len(tr) == 3000 and len(te) == 27000, (len(tr), len(te))
assert len(set(tr.tolist()) & set(te.tolist())) == 0, '训练/测试试次有重叠!'
print('    训练试次 %d, 测试试次 %d, 无重叠 ✓' % (len(tr), len(te)))

# ---------- 3) 标签: 试次 → 73k 图索引 ----------
print('[3] 读取 试次→图片 标签 ...')
with h5py.File(os.path.join(DATA, 'COCO_73k_subj_indices.hdf5'), 'r') as h:
    assert 'subj01' in h, list(h.keys())
    trial_img = np.asarray(h['subj01'][:], dtype=np.int64)      # [30000]
assert trial_img.shape == (30000,)
print('    标签范围 [%d, %d], 唯一图数 %d' %
      (trial_img.min(), trial_img.max(), len(np.unique(trial_img))))

# ---------- 4) Z-score: 用训练试次统计(防泄漏) ----------
print('[4] 按训练试次统计 z-score 每个 ROI ...')
mean = roi[tr].mean(axis=0)                                 # [100]
std = roi[tr].std(axis=0) + 1e-8
node_feat = ((roi - mean) / std).astype(np.float32)         # [30000,100]
np.savez(os.path.join(OUT, 'train_stats.npz'), mean=mean, std=std)
print('    z-score 后: 训练集 mean=%.4f std=%.4f' %
      (node_feat[tr].mean(), node_feat[tr].std()))

# ---------- 5) 固定邻接: 跨训练试次相关, >0.3 阈值(无abs, 与仓库一致) ----------
print('[5] 跨训练试次相关 → 固定邻接 ...')
X = node_feat[tr].astype(np.float64)                        # [3000,100]
Xc = X - X.mean(axis=0, keepdims=True)
cov = Xc.T @ Xc
d = np.sqrt(np.diag(cov))[:, None]
corr = cov / (d @ d.T + 1e-8)
corr = np.nan_to_num(corr, nan=0.0)
np.fill_diagonal(corr, 1.0)
adj = (corr > 0.3).astype(np.float32)
np.fill_diagonal(adj, 1.0)                                  # 自环(=BrainGFM 原始 adj 的行为)
print('    邻接密度: %.2f%%  (对角=1, 无负边)' % (adj.mean() * 100))

# ---------- 6) 保存 ----------
print('[6] 保存到', OUT, '...')
np.save(os.path.join(OUT, 'node_features.npy'), node_feat[:, :, None])   # [30000,100,1]
np.save(os.path.join(OUT, 'adj.npy'), adj)
np.save(os.path.join(OUT, 'trial_img.npy'), trial_img)
np.save(os.path.join(OUT, 'train_trials.npy'), tr)
np.save(os.path.join(OUT, 'test_trials.npy'), te)
print('    node_features.npy', node_feat[:, :, None].shape,
      ' adj.npy', adj.shape, ' trial_img.npy', trial_img.shape)

# ---------- 7) sanity ----------
tr_imgs = len(set(trial_img[tr].tolist()))
te_imgs = len(set(trial_img[te].tolist()))
print('[7] 划分检查: 训练图数=%d(应≈1000) 测试图数=%d(应≈9000)' % (tr_imgs, te_imgs))
print('    训练/测试图集无重叠:', len(set(trial_img[tr].tolist()) & set(trial_img[te].tolist())) == 0)
print('✅ 数据集构建完成。下一步: python 02_train_retrieval.py')
