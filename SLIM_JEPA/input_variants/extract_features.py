# -*- coding: utf-8 -*-
"""
extract_features.py  (第5章)
用 SLIM 编码器(修改版 hiera + best_model.pth)给每个试次提取特征向量。
先用 5 个试次冒烟测试(确认 forward 能跑、形状对),再全量提取。

用法:  python extract_features.py
输出:  feats_train.npy  (n_train, 512)
        feats_test.npy   (n_test, 512)
"""
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ============ 路径 ============
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / 'preprocess'))
sys.path.insert(0, str(PROJECT_DIR / 'model'))
from config import CHECKPOINT_PATH, DATA_DIR, NSD_MIN_DIR
from nsd_dataset import NSDSingleTrial
from hiera.hiera_mae import SlimEncoder

BETA = str(NSD_MIN_DIR / 'betas_all_subj01_fp32_renorm.hdf5')
MASK_F = str(NSD_MIN_DIR / 'nsdgeneral.nii.gz')
CKPT = str(CHECKPOINT_PATH)
DATA = str(DATA_DIR)

# ============ 模型 ============
cfg = dict(
    input_size=(40, 96, 96, 96), in_chans=1,
    patch_kernel=(1, 4, 4, 4), patch_stride=(1, 4, 4, 4), patch_padding=(0, 0, 0, 0),
    embed_dim=64, num_heads=1, stages=(2, 3, 16, 3), q_pool=2,
    q_stride=(2, 2, 2, 2), mask_unit_size=(8, 8, 8, 8), mlp_ratio=4.0, sep_pos_embed=True,
)
model = SlimEncoder(**cfg)
model.load_from_mae(CKPT)
model.eval()

# ============ 前景掩码(试次无关,用第一个样本算一次) ============
tmp_ds = NSDSingleTrial(BETA, MASK_F, trial_indices=[0], t_frames=40)
vol = tmp_ds[0][0].numpy()                                # (96,96,96) 第一帧
vol_t = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)   # (1,1,96,96,96)
with torch.no_grad():
    token_max = F.max_pool3d(vol_t, kernel_size=4, stride=4)      # (1,1,24,24,24)
    token_fore = token_max > 0
    mu = token_fore.view(1, 1, 3, 8, 3, 8, 3, 8).amax(dim=(3, 5, 7))  # (1,1,3,3,3)
    spatial_keep = mu.flatten().bool()                             # 27
mask = spatial_keep.repeat(5).unsqueeze(0)                         # (1, 135)
print('前景mask unit数:', int(mask.sum()), '/ 135')
USE_GPU = torch.cuda.is_available()
if USE_GPU:
    model = model.cuda()
    mask = mask.cuda()
print('使用GPU:', USE_GPU)

# ============ 提取函数 ============
def extract(idx_file, out_file, batch=2, max_n=None):   # batch=2 避免显存OOM; 若还爆改成1
    idx = np.load(idx_file)
    if max_n is not None:
        idx = idx[:max_n]
    ds = NSDSingleTrial(BETA, MASK_F, idx, t_frames=40)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    feats = []
    t0 = time.time()
    with torch.no_grad():
        for i, x in enumerate(dl):
            x = x.unsqueeze(1)                                   # (B,1,40,96,96,96)
            if USE_GPU:
                x = x.cuda()
            m = mask.repeat(x.shape[0], 1)
            f = model.forward_features(x, m)                     # (B,512)
            feats.append(f.float().cpu().numpy())
            if (i + 1) % 100 == 0:
                print(f'  {i+1}/{len(dl)} 批, 用时{time.time()-t0:.0f}s', flush=True)
    feats = np.concatenate(feats, 0)
    np.save(out_file, feats)
    print('已保存', out_file, feats.shape)

# ============ 先冒烟测试 5 个试次 ============
print('--- 冒烟测试: 5 个试次 ---')
extract(f'{DATA}/train_idx.npy', f'{DATA}/_smoke.npy', max_n=5)
print('冒烟测试通过! forward 正常,特征形状见上。')
print('开始全量提取...')

# ============ 全量提取 ============
extract(f'{DATA}/train_idx.npy', f'{DATA}/feats_train.npy')
extract(f'{DATA}/test_idx.npy',  f'{DATA}/feats_test.npy')
print('全部完成! feats_train.npy 和 feats_test.npy 已生成。')
