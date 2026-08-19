# -*- coding: utf-8 -*-
"""
extract_features_rawtime_full.py  方案2 全量特征提取
逐session处理(每次只缓存该session的12个run, 防OOM) + 批处理SLIM + 断点续传。
输出 feats2_train.npy / feats2_test.npy, 顺序和方案1完全对齐。
用法:  CUDA_VISIBLE_DEVICES=1 python extract_features_rawtime_full.py
"""
import sys, os, time
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / 'preprocess'))
sys.path.insert(0, str(PROJECT_DIR / 'model'))
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from scipy.ndimage import affine_transform
from torch.utils.data import DataLoader
from RawTimeDataset import RawTimeDataset
from config import CHECKPOINT_PATH, DATA_DIR, NSD_MIN_DIR, RAW_BOLD_DIR
from hiera.hiera_mae import SlimEncoder

DATA = str(DATA_DIR)
RAW = str(RAW_BOLD_DIR)
NSD_MASK = str(NSD_MIN_DIR / 'nsdgeneral.nii.gz')
CKPT = str(CHECKPOINT_PATH)
TRIAL_TR = f'{DATA}/trial_to_tr.npy'
TRIAL_META = f'{DATA}/trial_meta.npy'
BATCH = 4

# 要处理的session(0-based)。默认全部40。
# 中量测试只跑完整session: SESSIONS="0,1,2,4,5,6,7,8,9,10" (即session1-11, 跳过不完整的4)
_S = os.environ.get('SESSIONS', '')
SESSIONS = [int(x) for x in _S.split(',') if x.strip()] if _S else list(range(40))
print('本次处理的session:', [s + 1 for s in SESSIONS])

cfg = dict(input_size=(40,96,96,96), in_chans=1, patch_kernel=(1,4,4,4),
    patch_stride=(1,4,4,4), patch_padding=(0,0,0,0), embed_dim=64, num_heads=1,
    stages=(2,3,16,3), q_pool=2, q_stride=(2,2,2,2), mask_unit_size=(8,8,8,8),
    mlp_ratio=4.0, sep_pos_embed=True)

print('加载模型...')
model = SlimEncoder(**cfg); model.load_from_mae(CKPT); model.eval().cuda()

# ============ 前景mask(和方案2子集一致: nsdgeneral区) ============
img = nib.load(NSD_MASK); d = img.get_fdata()
m96 = affine_transform((d==1).astype(np.float32), np.linalg.inv(img.affine[:3,:3])*2.0,
                       offset=np.linalg.inv(img.affine[:3,:3])@(-96-img.affine[:3,3]),
                       output_shape=(96,96,96), order=0, mode='constant', cval=0) > 0.5
sample = np.zeros((96,96,96), dtype=np.float32); sample[m96] = 1.0
vt = torch.from_numpy(sample).float().unsqueeze(0).unsqueeze(0).cuda()
with torch.no_grad():
    token_act = F.avg_pool3d(vt, kernel_size=4, stride=4)
    tf = token_act.abs() > 0
    mu = tf.view(1,1,3,8,3,8,3,8).amax(dim=(3,5,7))
    mask = mu.flatten().bool().repeat(5).unsqueeze(0).cuda()
print(f'前景mask unit: {int(mask.sum())}/135')

meta = np.load(TRIAL_META)

def extract_split(idx_file, out_file):
    idx = np.load(idx_file)
    n = len(idx)
    if os.path.exists(out_file):
        feats = np.load(out_file)
        done_n = int((feats != 0).all(axis=1).sum())
        print(f'续传: 已处理 {done_n}/{n}')
    else:
        feats = np.zeros((n, 512), dtype=np.float32)
        done_n = 0
    for s in SESSIONS:
        pos = np.where(meta[idx, 0] == s)[0]
        if len(pos) == 0:
            continue
        if (feats[pos] != 0).all():
            done_n += len(pos)
            print(f'  session {s+1}: 已完成, 跳过')
            continue
        # 注意: 若该session数据不完整(缺run), 下面会直接报FileNotFoundError —— 这正是要的:
        # 报错=数据不完整, 确保完整性, 不会静默产出残缺特征。
        ds = RawTimeDataset(RAW, TRIAL_TR, TRIAL_META, trial_indices=idx[pos],
                            window=8, repeat=40, nsd_mask_file=NSD_MASK)
        dl = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0)
        t0 = time.time(); coll = []
        with torch.no_grad():
            for x in dl:
                x = x.unsqueeze(1).cuda()                # (B,1,40,96,96,96)
                m = mask.repeat(x.shape[0], 1)
                coll.append(model.forward_features(x, m).float().cpu().numpy())
        feats[pos] = np.concatenate(coll, 0)
        done_n += len(pos)
        np.save(out_file, feats)                          # 每session存一次(断点续传)
        print(f'  session {s+1}: {len(pos)} 试次, 用时{time.time()-t0:.0f}s, 累计{done_n}/{n}', flush=True)
        del ds, dl, coll                                   # 释放该session缓存
    print('完成', out_file, feats.shape)

print('===== 训练试次 =====')
extract_split(f'{DATA}/train_idx.npy', f'{DATA}/feats2_train.npy')
print('===== 测试试次 =====')
extract_split(f'{DATA}/test_idx.npy', f'{DATA}/feats2_test.npy')
print('方案2 全量特征提取完成! feats2_train.npy / feats2_test.npy 已生成')
