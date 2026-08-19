# -*- coding: utf-8 -*-
"""
clip_embeddings.py   (第6章)
把 train/test 用到的图片(~10000张)编码成 CLIP 向量并缓存。

用法:  python clip_embeddings.py
输出:  clip_emb.npy           (10000, D)   每个唯一图片的CLIP嵌入
        clip_unique_idx.npy   (10000,)     对应的图片索引
        clip_dim.txt          D            实际输出维度(存下来供后面用)
"""
import os
import sys
from pathlib import Path
import numpy as np
import h5py
import torch
from PIL import Image
from open_clip import create_model_and_transforms

# ================= 配置 =================
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from config import DATA_DIR, NSD_MIN_DIR

DATA = str(NSD_MIN_DIR)
IMGIDX_FILE = str(DATA_DIR / 'trial_imgidx.npy')  # 每个betas行的图片索引
OUT_PREFIX = str(DATA_DIR / 'clip')

# ================= 1. 收集用到的唯一图片 =================
imgidx = np.load(IMGIDX_FILE)                        # (30000,)
unique_idx = np.unique(imgidx).astype(int)           # 去重后的图片索引
print('用到的唯一图片数:', len(unique_idx), ' 范围:', unique_idx.min(), unique_idx.max())

# ================= 2. 加载 CLIP 模型 (ViT-bigG-14, 和MindEye2一致) =================
print('加载 CLIP 模型 ViT-bigG-14 ...')
model, _, preprocess = create_model_and_transforms(
    'ViT-bigG-14', pretrained='laion2b_s39b_b160k')
model.eval()
model = model.cuda()

# 探测实际输出维度(不写死,避免模型解析不同导致的维度错配)
with torch.no_grad():
    probe = torch.zeros(1, 3, 224, 224).cuda()
    OUT_DIM = model.encode_image(probe).shape[1]
print('CLIP 模型加载完成, 实际输出维度:', OUT_DIM)

# ================= 3. 逐张编码 (整个循环都在文件打开状态下进行) =================
emb = np.zeros((len(unique_idx), OUT_DIM), dtype=np.float32)

with h5py.File(f'{DATA}/coco_images_224_float16.hdf5', 'r') as f:
    images = f['images']                              # (73000, 3, 224, 224) float16, 值域0~1
    with torch.no_grad():
        for p, img_idx in enumerate(unique_idx):
            img = images[img_idx]                     # (3,224,224) float16 0~1
            arr = (np.transpose(img, (1, 2, 0)) * 255).astype(np.uint8)   # -> (H,W,C) 0~255
            pil = Image.fromarray(arr)
            t = preprocess(pil).unsqueeze(0).cuda()   # (1,3,224,224)
            emb[p] = model.encode_image(t).float().cpu().numpy()
            if (p + 1) % 1000 == 0:
                print(f'  已编码 {p+1}/{len(unique_idx)}', flush=True)

# ================= 4. 保存缓存 =================
np.save(f'{OUT_PREFIX}_emb.npy', emb)
np.save(f'{OUT_PREFIX}_unique_idx.npy', unique_idx)
with open(f'{OUT_PREFIX}_dim.txt', 'w') as f:
    f.write(str(OUT_DIM))
print('完成!已保存:')
print('  ', f'{OUT_PREFIX}_emb.npy', emb.shape)
print('  ', f'{OUT_PREFIX}_unique_idx.npy', unique_idx.shape)
print('  ', f'{OUT_PREFIX}_dim.txt', '维度 =', OUT_DIM)
