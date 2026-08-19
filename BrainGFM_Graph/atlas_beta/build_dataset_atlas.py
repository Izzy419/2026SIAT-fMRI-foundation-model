# -*- coding: utf-8 -*-
"""
build_dataset_atlas.py
======================
用「MNI 图谱 -> BOLD 区域」映射 + betas，构建 BrainGFM 检索数据集。

和方案B的区别（我们的 atlas 方案）：
  方案B：节点特征 = roi_ts.npy [30000,100]，来自 k-means 100 个聚类；
  本脚本：节点特征 = betas [30000,15724] 在每个解剖图谱区域内的平均，
          AAL -> 35 区，Schaefer -> 26 区。

格式约定（对齐方案B，保证可公平对比，逻辑自己实现）：
  节点特征 = 区域平均 beta，z-score（用训练试次统计，防数据泄漏）
  邻接     = 跨训练试次相关 > 0.3（对角=1）
  标签     = COCO 图索引

输入（只读，不改别人的数据）：
  atlas_labels_bold_<atlas>.npy     [15724]        体素 -> 区域编号（convert 脚本产出）
  betas_all_subj01_fp32_renorm.hdf5 ['betas']       [30000,15724] 每试次 beta
  COCO_73k_subj_indices.hdf5        ['subj01']      [30000] 试次 -> 图片
  subj01_train/test_trial_indices.npy               训练/测试试次划分

输出（写本脚本所在目录 R2.0）：
  node_features_<atlas>.npy  [30000, N, 1]  float32
  adj_<atlas>.npy            [N, N]         float32
  trial_img.npy              [30000]        共享（和方案B相同）
  train_trials.npy           [3000]         共享
  test_trials.npy            [27000]        共享
  train_stats_<atlas>.npz                   z-score 统计量

用法：
  python build_dataset_atlas.py --atlas aal
  python build_dataset_atlas.py --atlas schaefer
"""
import argparse
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
BETAS = os.path.join(DATA, 'nsd_subj01_min', 'betas_all_subj01_fp32_renorm.hdf5')
COCO = os.path.join(DATA, 'COCO_73k_subj_indices.hdf5')
PROC = str(PROCESSED_DIR)

parser = argparse.ArgumentParser()
parser.add_argument('--atlas', choices=['aal', 'schaefer'], default='aal')
args = parser.parse_args()
ATLAS = args.atlas

LABELS = os.path.join(HERE, 'atlas_labels_bold_%s.npy' % ATLAS)
assert os.path.isfile(LABELS), '找不到 %s，先运行 convert_atlas_to_bold.py' % LABELS

# ---------- 1. 读体素->区域标签，得到有效区域编号 ----------
print('[1] 读图谱标签 ...')
labels = np.load(LABELS)                        # [15724]，0=脑外
region_ids = np.unique(labels[labels > 0])      # 非连续区域编号（AAL 用 4101..，Schaefer 用 1..）
n_regions = len(region_ids)
print('    %s: 有效区域 %d 个' % (ATLAS, n_regions))

# ---------- 2. 读 betas，按区域平均 -> 节点特征 ----------
print('[2] 读 betas，按区域平均 ...')
with h5py.File(BETAS, 'r') as h:
    betas = h['betas'][:]                        # [30000,15724] float32
print('    betas:', betas.shape, betas.dtype)

# 平均权重矩阵 W [15724, N]：每列 = 该区域体素的平均权重(1/体素数)
W = np.zeros((len(labels), n_regions), dtype=np.float32)
for i, rid in enumerate(region_ids):
    vox = labels == rid
    W[vox, i] = 1.0 / vox.sum()

roi = betas @ W                                  # [30000, N] float32
print('    区域平均后节点特征:', roi.shape)

# ---------- 3. 读训练/测试试次划分 ----------
print('[3] 读训练/测试试次划分 ...')
tr = np.load(os.path.join(PROC, 'subj01_train_trial_indices.npy')).astype(np.int64)
te = np.load(os.path.join(PROC, 'subj01_test_trial_indices.npy')).astype(np.int64)
assert len(tr) == 3000 and len(te) == 27000, (len(tr), len(te))
assert len(set(tr.tolist()) & set(te.tolist())) == 0, '训练/测试试次有重叠!'
print('    训练 %d / 测试 %d，无重叠 ✓' % (len(tr), len(te)))

# ---------- 4. 读 试次->图片 标签 ----------
print('[4] 读 试次->图片 标签 ...')
with h5py.File(COCO, 'r') as h:
    assert 'subj01' in h, list(h.keys())
    trial_img = np.asarray(h['subj01'][:], dtype=np.int64)   # [30000]
assert trial_img.shape == (30000,)
print('    唯一图片数 %d' % len(np.unique(trial_img)))

# ---------- 5. z-score（用训练试次统计，防泄漏） ----------
print('[5] z-score（训练统计） ...')
mean = roi[tr].mean(axis=0)                      # [N]
std = roi[tr].std(axis=0) + 1e-8
node_feat = ((roi - mean) / std).astype(np.float32)
np.savez(os.path.join(HERE, 'train_stats_%s.npz' % ATLAS), mean=mean, std=std)
print('    训练集 mean=%.4f std=%.4f' % (node_feat[tr].mean(), node_feat[tr].std()))

# ---------- 6. 固定邻接：跨训练试次相关 >0.3 ----------
print('[6] 固定邻接 ...')
X = node_feat[tr].astype(np.float64)             # [3000,N]
Xc = X - X.mean(axis=0, keepdims=True)
cov = Xc.T @ Xc
d = np.sqrt(np.diag(cov))[:, None]
corr = cov / (d @ d.T + 1e-8)
corr = np.nan_to_num(corr, nan=0.0)
np.fill_diagonal(corr, 1.0)
adj = (corr > 0.3).astype(np.float32)
np.fill_diagonal(adj, 1.0)
print('    邻接密度 %.2f%%' % (adj.mean() * 100))

# ---------- 7. 保存 ----------
print('[7] 保存 ...')
np.save(os.path.join(HERE, 'node_features_%s.npy' % ATLAS), node_feat[:, :, None])  # [30000,N,1]
np.save(os.path.join(HERE, 'adj_%s.npy' % ATLAS), adj)
np.save(os.path.join(HERE, 'trial_img.npy'), trial_img)
np.save(os.path.join(HERE, 'train_trials.npy'), tr)
np.save(os.path.join(HERE, 'test_trials.npy'), te)
print('    node_features_%s.npy' % ATLAS, node_feat[:, :, None].shape)
print('    adj_%s.npy' % ATLAS, adj.shape)
print('    trial_img/train_trials/test_trials 已保存（共享）')
print('✅ 完成。下一步: python train_atlas.py --atlas %s' % ATLAS)
