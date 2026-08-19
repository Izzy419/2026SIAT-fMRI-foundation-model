# -*- coding: utf-8 -*-
"""
RawTimeDataset.py  (方案2)
从原始4D BOLD 按试次截取时间窗 -> 96³重采样 -> 加nsdgeneral掩码 -> 重复到40帧。
和方案1同区域(视觉区)、同格式(40,96,96,96)。
"""
import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform
import torch
from torch.utils.data import Dataset


class RawTimeDataset(Dataset):
    def __init__(self, rawdir, trial_to_tr, trial_meta, trial_indices,
                 window=8, repeat=40, zscore=True, nsd_mask_file=None):
        """
        rawdir:       rawdata_sub01 目录
        trial_to_tr:  trial_to_tr.npy (30000,) 每个试次的全局起始TR
        trial_meta:   trial_meta.npy (30000,3) (session, run, slot)
        trial_indices: 要用的试次索引
        window:       截取几个TR (默认8 = 12.8s)
        repeat:       最终帧数 (默认40)
        zscore:       整run Z-score (SLIM输入必须)
        nsd_mask_file: nsdgeneral.nii.gz; 提供则mask到视觉区(和方案1可比)
        """
        self.rawdir = rawdir
        self.tr = np.load(trial_to_tr)
        self.meta = np.load(trial_meta)
        self.trial_indices = np.asarray(trial_indices)
        self.window = window
        self.repeat = repeat
        self.zscore = zscore
        self.TR_PER_RUN = 188
        self._cache = {}
        self._aff = {}
        self.nsd_mask = None
        if nsd_mask_file is not None:
            img = nib.load(nsd_mask_file)
            d = img.get_fdata()
            A = img.affine[:3, :3]; t = img.affine[:3, 3]
            mat = np.linalg.inv(A) * 2.0
            off = np.linalg.inv(A) @ (-96 - t)
            m96 = affine_transform((d == 1).astype(np.float32), mat, offset=off,
                                   output_shape=(96, 96, 96), order=0,
                                   mode='constant', cval=0)
            self.nsd_mask = (m96 > 0.5)
            print('nsdgeneral 96³ 掩码体素数:', int(self.nsd_mask.sum()))

    def _load_run(self, s, r):
        key = (s, r)
        if key not in self._cache:
            p = (f'{self.rawdir}/ses-nsd{s+1:02d}/func/'
                 f'sub-01_ses-nsd{s+1:02d}_task-nsdcore_run-{r+1:02d}_bold.nii.gz')
            img = nib.load(p)
            data = img.get_fdata().astype(np.float32)      # (120,120,84,TR)
            if self.zscore:
                mu = data.mean(axis=-1, keepdims=True)
                sd = data.std(axis=-1, keepdims=True)
                valid = sd > 0.1
                data = np.where(valid, (data - mu) / (sd + 1e-6), 0.0)
            self._cache[key] = data
            self._aff[key] = img.affine
        return self._cache[key], self._aff[key]

    def _resample_to_96(self, vol, affine):
        A = affine[:3, :3]
        t = affine[:3, 3]
        mat = np.linalg.inv(A) * 2.0
        off = np.linalg.inv(A) @ (-96 - t)
        return affine_transform(vol, mat, offset=off,
                                output_shape=(96, 96, 96),
                                order=1, mode='constant', cval=0)

    def __len__(self):
        return len(self.trial_indices)

    def __getitem__(self, i):
        t = int(self.trial_indices[i])
        s, r, slot = self.meta[t]
        global_tr = int(self.tr[t])
        run_start = (s * 12 + r) * self.TR_PER_RUN
        inrun_onset = global_tr - run_start

        bold, aff = self._load_run(s, r)                   # (120,120,84,TR)
        w = self.window
        window_vols = bold[..., inrun_onset:inrun_onset + w]   # (120,120,84,W)
        vol96 = np.stack(
            [self._resample_to_96(window_vols[..., k], aff) for k in range(w)],
            axis=-1)                                      # (96,96,96,W)
        if self.nsd_mask is not None:
            vol96 = vol96 * self.nsd_mask[..., None].astype(np.float32)  # 只留视觉区
        n_tile = int(np.ceil(self.repeat / w))
        vol40 = np.tile(vol96, (1, 1, 1, n_tile))[..., :self.repeat]  # (96,96,96,40)
        return torch.from_numpy(vol40).permute(3, 0, 1, 2)  # (40,96,96,96)
