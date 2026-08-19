# -*- coding: utf-8 -*-
"""
convert_atlas_to_bold.py
========================
自己实现的「MNI 图谱 -> BOLD 原生空间」脑图转换。

背景：
  AAL / Schaefer 图谱都定义在 MNI 标准模板空间，而 NSD 的 BOLD 数据在
  「原生功能空间」（81x104x83 @1.8mm）。两者坐标不一样，需要一个配准场把
  图谱贴到 BOLD 上。NSD 官方提供了 MNI-to-func1pt8.nii.gz 这个配准场。

配准场存储约定（我们自己诊断出来的）：
  场在每个 BOLD 体素处存一个 3 维向量 = 该体素对应的 MNI 坐标，但做了
  「非负化」处理：真正的 MNI 世界坐标(mm) = 场值 - (90, 126, 72)。
  其中 (90,126,72) 是 MNI 模板的角点偏移（MNI 坐标范围 x∈[-90,90]、
  y∈[-126,90]、z∈[-72,108]，加上这个偏移后就都变成非负了；9999=脑外）。

流程：
  1. 读 BOLD 脑掩膜 nsdgeneral，得到所有脑体素坐标（15724 个）
  2. 每个脑体素查配准场 -> 得到非负 MNI 坐标
  3. 减偏移 -> 得到真实 MNI 世界坐标(mm)
  4. 用图谱 affine 反变换，把 MNI 世界坐标转成图谱里的体素索引
  5. 采样图谱标签，得到 15724 体素 -> 区域编号 的映射
  6. 打印覆盖率，保存 atlas_labels_bold.npy

用法：
  python convert_atlas_to_bold.py --atlas aal       # AAL 图谱
  python convert_atlas_to_bold.py --atlas schaefer  # Schaefer 图谱
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates

# ------------------------- 路径 -------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from config import ATLAS_CACHE_DIR, NSD_MASK_PATH, WARP_PATH as CONFIG_WARP_PATH

# BOLD 脑掩膜（nsdgeneral，15724 个脑体素）
MASK_PATH = str(NSD_MASK_PATH)
# NSD 官方配准场（MNI -> func 1.8mm）
WARP_PATH = str(CONFIG_WARP_PATH)
# 图谱缓存目录（AAL 和 Schaefer 都已下载好）
ATLAS_CACHE = str(ATLAS_CACHE_DIR)

# MNI 角点偏移：把非负化的场值还原成真实 MNI 世界坐标
MNI_OFFSET = np.array([90.0, 126.0, 72.0])
# 场里 >=9000 的值表示「脑外无效」（原文件填 9999）
INVALID_FILL = 9000.0

# ------------------------- 解析参数 -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--atlas', choices=['aal', 'schaefer'], default='aal')
parser.add_argument('--out', default=None, help='输出 npy 路径（默认当前目录）')
args = parser.parse_args()

# ------------------------- 1. 读脑掩膜 -------------------------
print('[1] 读脑掩膜 ...')
mask = nib.load(MASK_PATH).get_fdata()
brain_ijk = np.array(np.where(mask == 1)).T   # (N,3) 体素索引 [i,j,k]
print('    脑体素数:', len(brain_ijk))

# ------------------------- 2. 读配准场 -------------------------
print('[2] 读配准场 ...')
warp = nib.load(WARP_PATH).get_fdata()        # (81,104,83,3)
print('    场 shape:', warp.shape)

# 每个脑体素对应的场值（非负 MNI 坐标）
field = warp[brain_ijk[:, 0], brain_ijk[:, 1], brain_ijk[:, 2], :]  # (N,3)

# ------------------------- 3. 还原真实 MNI 世界坐标 -------------------------
print('[3] 还原 MNI 世界坐标: MNI = 场值 - (90,126,72) ...')
valid = (field < INVALID_FILL).all(axis=1)    # 9999 = 脑外
mni_world = field - MNI_OFFSET                # (N,3) mm

# ------------------------- 4. 加载图谱 -------------------------
print('[4] 加载图谱:', args.atlas)
if args.atlas == 'aal':
    atlas_path = os.path.join(ATLAS_CACHE, 'aal_SPM12', 'aal', 'ROI_MNI_V4.nii')
else:
    atlas_path = os.path.join(
        ATLAS_CACHE, 'schaefer_2018',
        'Schaefer2018_100Parcels_7Networks_order_FSLMNI152_1mm.nii.gz'
    )
atlas_img = nib.load(atlas_path)
atlas_data = np.asanyarray(atlas_img.dataobj)  # 3D 标签图

# 有些图谱是 4D（每个区域一个体），argmax 转成 3D 标签
if atlas_data.ndim == 4:
    atlas_data = np.argmax(atlas_data, axis=3).astype(np.int64) + 1
    atlas_data[atlas_data <= 0] = 0
print('    图谱 shape:', atlas_data.shape, ' 区域数:', len(np.unique(atlas_data)) - 1)

# ------------------------- 5. MNI 世界坐标 -> 图谱体素索引 -> 标签 -------------------------
print('[5] 映射图谱标签 ...')
inv_aff = np.linalg.inv(atlas_img.affine)
hom = np.concatenate([mni_world, np.ones((len(mni_world), 1))], axis=1)
atlas_vox = (inv_aff @ hom.T).T[:, :3]        # (N,3) 图谱体素索引(浮点)

labels = map_coordinates(
    atlas_data, atlas_vox.T, order=0,
    mode='constant', cval=0
).astype(np.int64)
labels[~valid] = 0                            # 脑外 -> 无标签

# ------------------------- 6. 覆盖率 + 保存 -------------------------
print('[6] 覆盖率 ...')
n_labeled = int((labels > 0).sum())
print('    %d 个脑体素有图谱标签，覆盖率 = %.1f%%' % (
    n_labeled, 100.0 * n_labeled / len(labels)
))
print('    覆盖到的区域数 = %d' % len(np.unique(labels[labels > 0])))

out = args.out or ('atlas_labels_bold_%s.npy' % args.atlas)
np.save(out, labels)
print('    已保存 ->', out)
