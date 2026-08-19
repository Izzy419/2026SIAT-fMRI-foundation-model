# -*- coding: utf-8 -*-
"""
extract_features_segbeta_v4.py
方案3 v4: 原始BOLD -> run-wise z-score -> FIR + ridge GLM -> 8 lag beta
-> 每个lag重复5帧，按时间顺序组成40-frame输入 -> SLIM feature。

关键修复:
1. 用 np.repeat(..., 5, axis=-1)，不再使用 tile 周期性循环8个beta。
2. 输出 feats3v4_train.npy / feats3v4_test.npy，和训练脚本一一对应。
3. 检查 NaN/Inf、shape、session/run完整性，并按session断点保存。
4. 保持与原方案3相同的SLIM encoder、mask和FIR基本协议。

用法:
    CUDA_VISIBLE_DEVICES=1 python extract_features_segbeta_v4.py
可选:
    SESSIONS="0,1,2" CUDA_VISIBLE_DEVICES=1 python extract_features_segbeta_v4.py
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import sys, time
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / 'model'))
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from scipy.ndimage import affine_transform
from config import CHECKPOINT_PATH, DATA_DIR, NSD_MIN_DIR, RAW_BOLD_DIR
from hiera.hiera_mae import SlimEncoder

DATA = str(DATA_DIR)
RAW = str(RAW_BOLD_DIR)
NSD_MASK = str(NSD_MIN_DIR / 'nsdgeneral.nii.gz')
CKPT = str(CHECKPOINT_PATH)

NLAGS = 8
REPEAT_PER_LAG = 5
TR_PER_RUN = 188
RIDGE = 100.0
BATCH = 4
OUT_DIM = 512

_s = os.environ.get('SESSIONS', '')
SESSIONS = [int(x) for x in _s.split(',') if x.strip()] if _s else list(range(40))

print(f'方案3 v4: sessions={[s+1 for s in SESSIONS]}', flush=True)
print(f'NLAGS={NLAGS}, repeat_per_lag={REPEAT_PER_LAG}, RIDGE={RIDGE}, BATCH={BATCH}', flush=True)

cfg = dict(
    input_size=(40, 96, 96, 96), in_chans=1,
    patch_kernel=(1, 4, 4, 4), patch_stride=(1, 4, 4, 4),
    patch_padding=(0, 0, 0, 0), embed_dim=64, num_heads=1,
    stages=(2, 3, 16, 3), q_pool=2, q_stride=(2, 2, 2, 2),
    mask_unit_size=(8, 8, 8, 8), mlp_ratio=4.0, sep_pos_embed=True
)

print('加载 SLIM...', flush=True)
model = SlimEncoder(**cfg)
model.load_from_mae(CKPT)
model.eval().cuda()

# ---------- mask ----------
_img = nib.load(NSD_MASK)
_d = _img.get_fdata()
_A = _img.affine[:3, :3]
_t = _img.affine[:3, 3]
_mat = np.linalg.inv(_A) * 2.0
_off = np.linalg.inv(_A) @ (-96 - _t)
nsd_mask = affine_transform(
    (_d == 1).astype(np.float32), _mat, offset=_off,
    output_shape=(96, 96, 96), order=0, mode='constant', cval=0
) > 0.5

# 使用稳定的前景定义，而不是依赖某个 beta sample 的数值正负
sample = nsd_mask.astype(np.float32)
vt = torch.from_numpy(sample).float().unsqueeze(0).unsqueeze(0).cuda()
with torch.no_grad():
    token_act = F.avg_pool3d(vt, kernel_size=4, stride=4)
    tf = token_act > 0
    mu = tf.view(1, 1, 3, 8, 3, 8, 3, 8).amax(dim=(3, 5, 7))
    FG_MASK = mu.flatten().bool().repeat(5).unsqueeze(0).cuda()
print(f'前景mask unit数: {int(FG_MASK.sum())}/135', flush=True)

tr = np.load(f'{DATA}/trial_to_tr.npy')
meta = np.load(f'{DATA}/trial_meta.npy')


def load_run(session, run):
    p = (
        f'{RAW}/ses-nsd{session+1:02d}/func/'
        f'sub-01_ses-nsd{session+1:02d}_task-nsdcore_run-{run+1:02d}_bold.nii.gz'
    )
    img = nib.load(p)
    data = img.get_fdata().astype(np.float32)
    mu = data.mean(axis=-1, keepdims=True)
    sd = data.std(axis=-1, keepdims=True)
    valid = sd > 0.1
    data = np.where(valid, (data - mu) / (sd + 1e-6), 0.0)
    return data, img.affine


def fit_fir_glm(bold, onsets, nlags=NLAGS, ridge=RIDGE):
    """每个trial各有nlags个FIR系数；另加截距和二次时间趋势。"""
    T = bold.shape[-1]
    nt = len(onsets)
    n_reg = nt * nlags + 3
    X = np.zeros((T, n_reg), dtype=np.float32)
    for ti, on in enumerate(onsets):
        for lag in range(nlags):
            row = int(on) + lag
            if 0 <= row < T:
                X[row, ti * nlags + lag] = 1.0
    tt = np.arange(T, dtype=np.float32) / max(T, 1)
    X[:, nt * nlags] = 1.0
    X[:, nt * nlags + 1] = tt
    X[:, nt * nlags + 2] = tt ** 2

    # 在较小的时间维上直接解正则化正规方程
    A = np.linalg.solve(
        X.T @ X + ridge * np.eye(n_reg, dtype=np.float32),
        X.T
    )
    Y = bold.reshape(-1, T)
    betas = A @ Y.T
    return betas[:nt * nlags].reshape(nt, nlags, -1)


def to96_mask(vol, aff):
    A = aff[:3, :3]
    t = aff[:3, 3]
    v = affine_transform(
        vol, np.linalg.inv(A) * 2.0,
        offset=np.linalg.inv(A) @ (-96 - t),
        output_shape=(96, 96, 96), order=1,
        mode='constant', cval=0
    )
    return v * nsd_mask.astype(np.float32)


def make_40frame_from_betas(beta_lag_vols):
    """
    输入: (96,96,96,NLAGS)
    输出: (40,96,96,96)
    每个lag连续占5帧:
        lag0 x5, lag1 x5, ..., lag7 x5
    """
    v40 = np.repeat(beta_lag_vols, REPEAT_PER_LAG, axis=-1)
    assert v40.shape[-1] == 40, v40.shape
    return np.moveaxis(v40, -1, 0).astype(np.float32, copy=False)


def validate_existing(feats, n, path):
    if feats.shape != (n, OUT_DIM):
        raise ValueError(f'{path} shape={feats.shape}, expected={(n, OUT_DIM)}')
    bad = ~np.isfinite(feats).all(axis=1)
    if bad.any():
        raise ValueError(f'{path} 有 {int(bad.sum())} 行包含 NaN/Inf')


def extract_split(idx_file, out_file):
    idx = np.load(idx_file)
    n = len(idx)
    print(f'开始: {os.path.basename(idx_file)}, n={n}', flush=True)

    if os.path.exists(out_file):
        feats = np.load(out_file)
        validate_existing(feats, n, out_file)
        done_mask = (np.abs(feats).sum(axis=1) > 0)
        print(f'续传: {int(done_mask.sum())}/{n} 行已有特征', flush=True)
    else:
        feats = np.zeros((n, OUT_DIM), dtype=np.float32)
        done_mask = np.zeros(n, dtype=bool)

    for s in SESSIONS:
        pos = np.where(meta[idx, 0] == s)[0]
        if len(pos) == 0:
            continue
        if done_mask[pos].all():
            print(f'  session {s+1}: 已完成，跳过', flush=True)
            continue

        # 按 run 聚合 trial；保证同一run一次GLM
        run_groups = {}
        for k, tpos in enumerate(pos):
            t = int(idx[tpos])
            run_groups.setdefault(int(meta[t, 1]), []).append(k)

        batch_xs, batch_ks = [], []
        t0 = time.time()

        def flush():
            if not batch_xs:
                return
            xb = torch.stack(batch_xs).unsqueeze(1).cuda()
            m = FG_MASK.repeat(len(batch_xs), 1)
            with torch.no_grad():
                f = model.forward_features(xb, m).float().cpu().numpy()
            if f.shape[1] != OUT_DIM:
                raise ValueError(f'SLIM feature dim={f.shape[1]}, expected={OUT_DIM}')
            for j, k in enumerate(batch_ks):
                feats[pos[k]] = f[j]
                done_mask[pos[k]] = True
            batch_xs.clear(); batch_ks.clear()

        for r in sorted(run_groups):
            klist = run_groups[r]
            trials = idx[pos[klist]]
            onsets = (tr[trials] - (s * 12 + r) * TR_PER_RUN).astype(int)
            if ((onsets < 0) | (onsets >= TR_PER_RUN)).any():
                raise ValueError(f'session {s+1} run {r+1} 存在越界onset: {onsets.tolist()}')

            bold, aff = load_run(s, r)
            tbs = fit_fir_glm(bold, onsets)
            shape = bold.shape[:3]

            for ti, k in enumerate(klist):
                vols = tbs[ti].reshape(NLAGS, *shape)
                beta96 = np.stack(
                    [to96_mask(vols[lag], aff) for lag in range(NLAGS)], axis=-1
                )
                vol40 = make_40frame_from_betas(beta96)
                batch_xs.append(torch.from_numpy(vol40))
                batch_ks.append(k)
                if len(batch_xs) >= BATCH:
                    flush()

            del bold, tbs

        flush()
        np.save(out_file, feats)
        print(
            f'  session {s+1}: {len(pos)} trials, '
            f'{time.time()-t0:.0f}s, 累计{int(done_mask.sum())}/{n}',
            flush=True
        )

    validate_existing(feats, n, out_file)
    np.save(out_file, feats)
    print(f'完成 {out_file}: {feats.shape}', flush=True)


print('===== v4 训练特征 =====', flush=True)
extract_split(f'{DATA}/train_idx.npy', f'{DATA}/feats3v4_train.npy')
print('===== v4 测试特征 =====', flush=True)
extract_split(f'{DATA}/test_idx.npy', f'{DATA}/feats3v4_test.npy')
print('方案3 v4 特征提取完成。', flush=True)
