# -*- coding: utf-8 -*-
"""
probe_scaling_law_final.py

Probe 训练数据量幂律缩放实验（最终版）

实验协议
--------
全部 unique images = 10,000，每张图片恰好 3 个 trial。

固定 image-level 划分：
    Training pool : 6,000 images = 最多 18,000 trials
    Validation    : 1,000 images = 3,000 trials
    Test          : 3,000 images = 9,000 trials

Scaling points：
    100, 200, 400, 800, 1200, 2000, 3000, 4000, 5000, 6000 images
对应：
    300, 600, 1200, 2400, 3600, 6000, 9000, 12000, 15000, 18000 trials

关键原则
--------
1. train/val/test 在 image level 完全互斥；
2. 一张 image 的 3 个 trial 永远在同一个 split；
3. 所有 seed 共用同一个固定 split；
4. 所有 seed 共用同一个 nested training subset：
       N=100 ⊂ 200 ⊂ 400 ⊂ ... ⊂ 6000
   seed 只改变：
       - projector 初始化
       - train image-group 的 batch shuffle
5. encoder 完全冻结，只预计算一次全部 30,000 trial 的 512-d feature；
6. 同一张 image 的 3 个 trial 在训练 batch 中保持成组，
   因此 multi-positive InfoNCE 能看到全部 3 个正样本；
7. validation 只用于选择 best checkpoint；
8. test 只在 best checkpoint 上评估，固定不变；
9. 最后自动拟合：
       Accuracy(N) = C - A * N^(-alpha)
   并输出 alpha。

运行
----
首次准备：
    CUDA_VISIBLE_DEVICES=1 python probe_scaling_law_final.py --prepare-only

正式运行：
    CUDA_VISIBLE_DEVICES=1 nohup python probe_scaling_law_final.py \
        > probe_scaling_final.log 2>&1 &

默认每个 scale 运行 3 seeds。
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

import sys
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "preprocess"))
sys.path.insert(0, str(PROJECT_DIR / "model"))
from config import CHECKPOINT_PATH, DATA_DIR, NSD_MIN_DIR
from nsd_dataset import NSDSingleTrial
from hiera.hiera_mae import SlimEncoder


# ============================================================
# Paths
# ============================================================

DATA = DATA_DIR

BETA = NSD_MIN_DIR / "betas_all_subj01_fp32_renorm.hdf5"
MASK_F = NSD_MIN_DIR / "nsdgeneral.nii.gz"
CKPT = CHECKPOINT_PATH

FEATURE_FILE = DATA / "probe_scaling_all_features.npy"
SPLIT_FILE = DATA / "probe_scaling_split_6000.npz"
RESULT_FILE = DATA / "mindeye2_scaling_results_v2.csv"
FIT_FILE = DATA / "probe_scaling_powerlaw_fit.txt"
MODEL_DIR = DATA / "probe_scaling_models_final"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Experiment settings
# ============================================================

TRAIN_POOL_IMAGES = 6000
VAL_IMAGES = 1000
TEST_IMAGES = 3000

DEFAULT_SCALES = [
    100, 200, 400, 800, 1200,
    2000, 3000, 4000, 5000, 6000
]

DEFAULT_SEEDS = [0, 1, 2]

EPOCHS = 150
BATCH_IMAGES = 40          # 40 images × 3 trials = 120 trials / batch
LR = 3e-4
WEIGHT_DECAY = 0.05
TEMP = 0.1
VAL_EVERY = 10

FEATURE_BATCH = 4


# ============================================================
# Slim config
# ============================================================

CFG = dict(
    input_size=(40, 96, 96, 96),
    in_chans=1,
    patch_kernel=(1, 4, 4, 4),
    patch_stride=(1, 4, 4, 4),
    patch_padding=(0, 0, 0, 0),
    embed_dim=64,
    num_heads=1,
    stages=(2, 3, 16, 3),
    q_pool=2,
    q_stride=(2, 2, 2, 2),
    mask_unit_size=(8, 8, 8, 8),
    mlp_ratio=4.0,
    sep_pos_embed=True,
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Data helpers
# ============================================================

def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def load_core_data():
    trial_imgidx = np.load(DATA / "trial_imgidx.npy").astype(np.int64)
    clip_emb = np.load(DATA / "clip_emb.npy").astype(np.float32)
    clip_uidx = np.load(DATA / "clip_unique_idx.npy").astype(np.int64)

    return trial_imgidx, clip_emb, clip_uidx


def validate_all_images(trial_imgidx: np.ndarray) -> np.ndarray:
    images, counts = np.unique(trial_imgidx, return_counts=True)

    bad = images[counts != 3]
    if len(bad):
        raise RuntimeError(
            f"发现 {len(bad)} 张图片不是恰好 3 个 trial，"
            f"例如 {bad[:10].tolist()}"
        )

    print(
        f"全部 unique images = {len(images)}, "
        f"全部 trials = {len(trial_imgidx)}, "
        f"每张图片 = 3 trials",
        flush=True,
    )

    return images.astype(np.int64)


def images_to_trial_indices(
    trial_imgidx: np.ndarray,
    images: np.ndarray,
) -> np.ndarray:
    wanted = np.asarray(images, dtype=np.int64)
    mask = np.isin(trial_imgidx, wanted)
    return np.flatnonzero(mask).astype(np.int64)


def make_fixed_split(
    unique_images: np.ndarray,
    seed: int = 2026,
) -> dict:
    required = TRAIN_POOL_IMAGES + VAL_IMAGES + TEST_IMAGES

    if len(unique_images) != required:
        raise RuntimeError(
            f"本脚本预期 exactly {required} unique images，"
            f"但当前发现 {len(unique_images)}。"
        )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(unique_images)

    test_images = perm[:TEST_IMAGES]

    val_images = perm[
        TEST_IMAGES:
        TEST_IMAGES + VAL_IMAGES
    ]

    train_pool_images = perm[
        TEST_IMAGES + VAL_IMAGES:
        TEST_IMAGES + VAL_IMAGES + TRAIN_POOL_IMAGES
    ]

    return {
        "test_images": test_images.astype(np.int64),
        "val_images": val_images.astype(np.int64),
        "train_pool_images": train_pool_images.astype(np.int64),
    }


def save_fixed_split(split: dict) -> None:
    np.savez(
        SPLIT_FILE,
        test_images=split["test_images"],
        val_images=split["val_images"],
        train_pool_images=split["train_pool_images"],
    )

    print(
        f"固定 split 已保存: {SPLIT_FILE}",
        flush=True,
    )


def load_fixed_split() -> dict:
    z = np.load(SPLIT_FILE)

    return {
        "test_images": z["test_images"].astype(np.int64),
        "val_images": z["val_images"].astype(np.int64),
        "train_pool_images": z["train_pool_images"].astype(np.int64),
    }


# ============================================================
# Frozen encoder feature extraction
# ============================================================

@torch.no_grad()
def build_foreground_mask() -> torch.Tensor:
    ds = NSDSingleTrial(
        BETA,
        MASK_F,
        [0],
        t_frames=40,
    )

    vol = ds[0][0].numpy()

    vt = (
        torch.from_numpy(vol)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
        .cuda()
    )

    token_max = F.max_pool3d(
        vt,
        kernel_size=4,
        stride=4,
    )

    token_fore = token_max > 0

    mu = token_fore.view(
        1, 1, 3, 8, 3, 8, 3, 8
    ).amax(dim=(3, 5, 7))

    mask = (
        mu.flatten()
        .bool()
        .repeat(5)
        .unsqueeze(0)
        .cuda()
    )

    del ds, vt, token_max, token_fore, mu

    return mask


@torch.no_grad()
def precompute_all_features(
    trial_imgidx: np.ndarray,
) -> None:

    expected_shape = (
        len(trial_imgidx),
        512,
    )

    if FEATURE_FILE.exists():
        arr = np.load(
            FEATURE_FILE,
            mmap_mode="r",
        )

        if arr.shape == expected_shape:
            print(
                f"已有 frozen features，直接复用: "
                f"{FEATURE_FILE}, shape={arr.shape}",
                flush=True,
            )
            return

        print(
            f"已有 feature 文件形状 {arr.shape} "
            f"与预期 {expected_shape} 不符，将重新生成。",
            flush=True,
        )

    print(
        "开始预计算全部 trial 的 frozen-encoder features...",
        flush=True,
    )

    encoder = SlimEncoder(**CFG)
    encoder.load_from_mae(CKPT)
    encoder.eval().cuda()

    mask = build_foreground_mask()

    ds = NSDSingleTrial(
        BETA,
        MASK_F,
        np.arange(
            len(trial_imgidx),
            dtype=np.int64,
        ),
        t_frames=40,
    )

    feats = np.zeros(
        expected_shape,
        dtype=np.float32,
    )

    t0 = time.time()

    for start in range(
        0,
        len(ds),
        FEATURE_BATCH,
    ):
        end = min(
            start + FEATURE_BATCH,
            len(ds),
        )

        xs = []

        for j in range(start, end):
            item = ds[j]

            x = (
                item[0]
                if isinstance(item, (tuple, list))
                else item
            )

            xs.append(
                torch.from_numpy(
                    np.asarray(x)
                ).float()
            )

        x = (
            torch.stack(xs)
            .unsqueeze(1)
            .cuda()
        )

        m = mask.repeat(
            x.shape[0],
            1,
        )

        f = encoder.forward_features(
            x,
            m,
        )

        if f.ndim != 2 or f.shape[1] != 512:
            raise RuntimeError(
                f"encoder feature shape 异常: {tuple(f.shape)}"
            )

        feats[start:end] = (
            f.float()
            .cpu()
            .numpy()
        )

        if end % 500 == 0 or end == len(ds):
            print(
                f"  {end}/{len(ds)} "
                f"({100 * end / len(ds):.1f}%), "
                f"{time.time() - t0:.0f}s",
                flush=True,
            )

        del x, m, f, xs

    np.save(
        FEATURE_FILE,
        feats,
    )

    print(
        f"frozen features 完成: "
        f"{FEATURE_FILE}, shape={feats.shape}",
        flush=True,
    )

    del encoder, ds, feats, mask
    torch.cuda.empty_cache()


# ============================================================
# Grouped sampler
# ============================================================

class ImageGroupBatchSampler(Sampler):
    """
    保证一个 batch 由完整的 image groups 构成：
        BATCH_IMAGES images × 3 trials

    这样同一 image 的 3 个 trial 一定同时进入 batch，
    multi-positive InfoNCE 才能看到全部 3 个正样本。
    """

    def __init__(
        self,
        image_to_trial_positions: list[list[int]],
        batch_images: int,
        seed: int,
        epoch: int = 0,
    ):
        self.groups = image_to_trial_positions
        self.batch_images = batch_images
        self.seed = seed
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(
            self.seed + self.epoch
        )

        order = rng.permutation(
            len(self.groups)
        )

        for start in range(
            0,
            len(order),
            self.batch_images,
        ):
            chosen = order[
                start:
                start + self.batch_images
            ]

            batch = []

            for gi in chosen:
                batch.extend(
                    self.groups[int(gi)]
                )

            if batch:
                yield batch

    def __len__(self):
        return math.ceil(
            len(self.groups) / self.batch_images
        )


class GroupedTrainDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        trial_imgids: np.ndarray,
    ):
        self.features = features
        self.trial_imgids = trial_imgids

        unique = np.unique(
            self.trial_imgids
        )

        self.groups = []

        for img in unique:
            positions = np.flatnonzero(
                self.trial_imgids == img
            ).tolist()

            if len(positions) != 3:
                raise RuntimeError(
                    f"image {int(img)} has "
                    f"{len(positions)} trials, expected 3"
                )

            self.groups.append(
                positions
            )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(
                np.asarray(
                    self.features[idx]
                )
            ).float(),
            int(self.trial_imgids[idx]),
        )


# ============================================================
# Probe model / loss
# ============================================================

class ClipProjector(nn.Module):
    """方案二: 忠实复现 MindEye2 BrainNetwork 主干(MLP-Mixer: token mixing + channel mixing)"""
    def __init__(
        self,
        in_dim=512,
        out_dim=1280,
        h=1024,
        seq_len=1,
        n_blocks=4,
        drop=0.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.h = h

        # 输入投影(对应 MindEye2 的被试线性层)
        self.in_proj = nn.Linear(in_dim, h * seq_len)

        # MLP-Mixer blocks(忠实复现 mixer_block1/2)
        self.mixer_blocks1 = nn.ModuleList(
            [self._mixer_block1(h, drop) for _ in range(n_blocks)]
        )
        self.mixer_blocks2 = nn.ModuleList(
            [self._mixer_block2(seq_len, drop) for _ in range(n_blocks)]
        )

        self.backbone_linear = nn.Linear(h * seq_len, out_dim)

    def _mlp(self, i, o, drop):
        return nn.Sequential(
            nn.Linear(i, o),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(o, o),
        )

    def _mixer_block1(self, h, drop):
        # token mixing(沿 h 维度)
        return nn.Sequential(
            nn.LayerNorm(h),
            self._mlp(h, h, drop),
        )

    def _mixer_block2(self, s, drop):
        # channel mixing(沿 seq_len 维度)
        return nn.Sequential(
            nn.LayerNorm(s),
            self._mlp(s, s, drop),
        )

    def forward(self, x):
        x = self.in_proj(x).reshape(len(x), self.seq_len, self.h)
        residual1 = x
        residual2 = x.permute(0, 2, 1)
        for block1, block2 in zip(self.mixer_blocks1, self.mixer_blocks2):
            x = block1(x) + residual1
            residual1 = x
            x = x.permute(0, 2, 1)
            x = block2(x) + residual2
            residual2 = x
            x = x.permute(0, 2, 1)
        x = x.reshape(len(x), -1)
        return F.normalize(
            self.backbone_linear(x),
            dim=-1,
        )


def multipositive_nce(
    q: torch.Tensor,
    k: torch.Tensor,
    image_ids: torch.Tensor,
) -> torch.Tensor:

    logits = (
        q @ k.T
    ) / TEMP

    ids = image_ids.view(-1)

    positive_mask = (
        ids[:, None]
        == ids[None, :]
    )

    positive_logits = logits.masked_fill(
        ~positive_mask,
        float("-inf"),
    )

    return -(
        torch.logsumexp(
            positive_logits,
            dim=1,
        )
        - torch.logsumexp(
            logits,
            dim=1,
        )
    ).mean()


# ============================================================
# Retrieval
# ============================================================

@torch.no_grad()
def encode_features(
    model,
    features,
) -> np.ndarray:

    model.eval()

    x = (
        torch.from_numpy(
            np.asarray(features)
        )
        .float()
        .cuda()
    )

    q = (
        model(x)
        .cpu()
        .numpy()
    )

    return q


def evaluate_100way(
    q: np.ndarray,
    img_ids: np.ndarray,
    clip_n: np.ndarray,
    id2pos: dict[int, int],
    candidate_images: np.ndarray,
):
    candidate_images = np.asarray(
        candidate_images,
        dtype=np.int64,
    )

    rng = np.random.default_rng(0)

    t1 = 0
    t5 = 0
    t10 = 0
    one = 0

    for i in range(len(q)):
        gt = int(img_ids[i])

        if gt not in id2pos:
            raise RuntimeError(
                f"image {gt} 没有 CLIP embedding"
            )

        if gt not in candidate_images:
            raise RuntimeError(
                f"test sample image {gt} "
                f"不在 candidate image set"
            )

        others = candidate_images[
            candidate_images != gt
        ]

        if len(others) < 99:
            raise RuntimeError(
                "100-way requires至少99个 distractors"
            )

        distractors = rng.choice(
            others,
            size=99,
            replace=False,
        )

        pool_ids = np.concatenate(
            [
                np.asarray(
                    [gt],
                    dtype=np.int64,
                ),
                distractors,
            ]
        )

        pool_pos = np.asarray(
            [
                id2pos[int(x)]
                for x in pool_ids
            ],
            dtype=np.int64,
        )

        scores = (
            clip_n[pool_pos]
            @ q[i]
        )

        rank_order = np.argsort(
            -scores
        )

        gt_rank = int(
            np.where(
                rank_order == 0
            )[0][0]
        )

        t1 += int(gt_rank < 1)
        t5 += int(gt_rank < 5)
        t10 += int(gt_rank < 10)

        d1 = rng.choice(others)

        pair_pos = np.asarray(
            [
                id2pos[int(gt)],
                id2pos[int(d1)],
            ],
            dtype=np.int64,
        )

        one += int(
            np.argmax(
                clip_n[pair_pos] @ q[i]
            ) == 0
        )

    n = len(q)

    return (
        t1 / n,
        t5 / n,
        t10 / n,
        one / n,
    )


# ============================================================
# Train one scaling point
# ============================================================

def train_one_scale(
    train_features,
    train_imgids,
    val_features,
    val_imgids,
    clip_n,
    id2pos,
    val_candidates,
    seed,
    epochs,
):
    set_seed(seed)

    model = ClipProjector().cuda()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # CLIP target：先从原始 image ID 映射到 clip_unique_idx 的位置
    train_clip_pos = np.asarray(
        [
            id2pos[int(x)]
            for x in train_imgids
        ],
        dtype=np.int64,
    )

    Y = normalize_rows(
        clip_n[train_clip_pos]
    ).astype(np.float32)

    train_ds = GroupedTrainDataset(
        train_features,
        train_imgids,
    )

    # 根据 Dataset 中的 image groups 建立 batch sampler
    sampler = ImageGroupBatchSampler(
        train_ds.groups,
        batch_images=BATCH_IMAGES,
        seed=seed,
    )

    loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        num_workers=0,
    )

    # Y 与 features 完全同顺序，因此把 target 直接放 Dataset 不必要；
    # 这里在训练 loop 按 image ID 动态查 CLIP target。
    total_steps = max(
        1,
        epochs * len(loader),
    )

    warmup_steps = max(
        1,
        5 * len(loader),
    )

    def lr_lambda(step):
        if step < warmup_steps:
            return (
                step
                / warmup_steps
            )

        return 0.5 * (
            1.0
            + math.cos(
                math.pi
                * (
                    step
                    - warmup_steps
                )
                / max(
                    1,
                    total_steps
                    - warmup_steps,
                )
            )
        )

    scheduler = (
        torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda,
        )
    )

    best_val_top1 = -1.0
    best_state = None

    # 用 image ID → CLIP row 的直接映射，训练时查
    for epoch in range(epochs):

        # 每个 epoch 都重新打乱 image groups
        loader = DataLoader(
            train_ds,
            batch_sampler=ImageGroupBatchSampler(
                train_ds.groups,
                batch_images=BATCH_IMAGES,
                seed=seed,
                epoch=epoch,
            ),
            num_workers=0,
        )

        model.train()

        loss_sum = 0.0
        samples = 0

        for xb, ib in loader:

            xb = xb.cuda()

            ib = ib.numpy()

            clip_positions = np.asarray(
                [
                    id2pos[int(x)]
                    for x in ib
                ],
                dtype=np.int64,
            )

            yb = torch.from_numpy(
                clip_n[clip_positions]
            ).float().cuda()

            q = model(xb)

            loss = multipositive_nce(
                q,
                yb,
                torch.from_numpy(ib).long().cuda(),
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()
            scheduler.step()

            bs = xb.shape[0]
            loss_sum += (
                loss.item()
                * bs
            )

            samples += bs

        train_loss = (
            loss_sum
            / max(1, samples)
        )

        if (
            (epoch + 1)
            % VAL_EVERY
            == 0
            or epoch == epochs - 1
        ):
            val_q = encode_features(
                model,
                val_features,
            )

            v1, v5, v10, v1way = (
                evaluate_100way(
                    val_q,
                    val_imgids,
                    clip_n,
                    id2pos,
                    val_candidates,
                )
            )

            print(
                f"seed={seed} "
                f"epoch={epoch + 1}/{epochs} "
                f"loss={train_loss:.5f} "
                f"valTop1={v1:.5f} "
                f"valTop5={v5:.5f} "
                f"valTop10={v10:.5f} "
                f"val1way={v1way:.5f}",
                flush=True,
            )

            if v1 > best_val_top1:
                best_val_top1 = v1

                best_state = {
                    k: v.detach()
                    .cpu()
                    .clone()
                    for k, v
                    in model.state_dict().items()
                }

    if best_state is None:
        raise RuntimeError(
            "没有成功保存 validation best checkpoint"
        )

    model.load_state_dict(
        best_state
    )

    return (
        model,
        best_val_top1,
    )


# ============================================================
# Power-law fitting
# ============================================================

def fit_power_law(
    n,
    y,
):
    """
    y(N) = C - A * N^(-alpha)

    用 scipy curve_fit（若环境有 scipy）；
    没有 scipy 时退化为简单网格搜索。
    """

    n = np.asarray(n, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    finite = (
        np.isfinite(n)
        & np.isfinite(y)
        & (n > 0)
    )

    n = n[finite]
    y = y[finite]

    if len(n) < 4:
        return None

    try:
        from scipy.optimize import curve_fit

        def func(N, C, A, alpha):
            return (
                C
                - A
                * np.power(N, -alpha)
            )

        c0 = float(
            max(y)
            + 0.01
        )

        a0 = float(
            max(
                1e-4,
                c0 - min(y),
            )
        )

        p0 = [
            c0,
            a0,
            0.2,
        ]

        bounds = (
            [
                -1.0,
                0.0,
                0.001,
            ],
            [
                2.0,
                10.0,
                5.0,
            ],
        )

        popt, _ = curve_fit(
            func,
            n,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=100000,
        )

        pred = func(
            n,
            *popt,
        )

        ss_res = np.sum(
            (y - pred) ** 2
        )

        ss_tot = np.sum(
            (y - y.mean()) ** 2
        )

        r2 = (
            1.0
            - ss_res
            / max(
                ss_tot,
                1e-12,
            )
        )

        return {
            "C": float(popt[0]),
            "A": float(popt[1]),
            "alpha": float(popt[2]),
            "R2": float(r2),
        }

    except Exception as e:
        print(
            f"power-law fit failed: {e}",
            flush=True,
        )
        return None


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
    )

    parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=DEFAULT_SCALES,
    )

    parser.add_argument(
        "--prepare-only",
        action="store_true",
    )

    args = parser.parse_args()

    print("=" * 72)
    print(
        "MindEye2 Head Training-Data Scaling Law — FINAL"
    )
    print("=" * 72)

    print(
        f"scales={args.scales}"
    )

    print(
        f"seeds={args.seeds}"
    )

    print(
        f"epochs={args.epochs}"
    )

    print(
        f"train pool={TRAIN_POOL_IMAGES} images "
        f"/ {TRAIN_POOL_IMAGES * 3} trials"
    )

    print(
        f"validation={VAL_IMAGES} images "
        f"/ {VAL_IMAGES * 3} trials"
    )

    print(
        f"test={TEST_IMAGES} images "
        f"/ {TEST_IMAGES * 3} trials"
    )

    print()

    # --------------------------------------------------------
    # Load core arrays
    # --------------------------------------------------------

    trial_imgidx, clip_emb, clip_uidx = (
        load_core_data()
    )

    unique_images = (
        validate_all_images(
            trial_imgidx
        )
    )

    clip_n = normalize_rows(
        clip_emb
    ).astype(np.float32)

    id2pos = {
        int(image_id): int(pos)
        for pos, image_id
        in enumerate(clip_uidx)
    }

    missing = [
        int(img)
        for img in unique_images
        if int(img) not in id2pos
    ]

    if missing:
        raise RuntimeError(
            f"有 {len(missing)} 张 image 没有 CLIP embedding，"
            f"例如 {missing[:10]}"
        )

    # --------------------------------------------------------
    # Fixed split
    # --------------------------------------------------------

    if SPLIT_FILE.exists():

        split = load_fixed_split()

        print(
            f"加载固定 split: {SPLIT_FILE}",
            flush=True,
        )

    else:

        split = make_fixed_split(
            unique_images,
            seed=2026,
        )

        save_fixed_split(
            split
        )

    # 验证 split 不重叠
    train_set = set(
        split["train_pool_images"]
    )

    val_set = set(
        split["val_images"]
    )

    test_set = set(
        split["test_images"]
    )

    if (
        train_set & val_set
        or train_set & test_set
        or val_set & test_set
    ):
        raise RuntimeError(
            "train/val/test image split 有重叠"
        )

    if len(
        train_set
        | val_set
        | test_set
    ) != 10000:
        raise RuntimeError(
            "train/val/test 没有覆盖全部 10000 images"
        )

    # --------------------------------------------------------
    # Nested training subset
    #
    # 所有 seeds 使用同一套 image ordering。
    # 这样 seed 不再改变数据子集，只改变训练随机性。
    # --------------------------------------------------------

    split_rng = np.random.default_rng(
        2027
    )

    nested_train_order = (
        split_rng.permutation(
            split["train_pool_images"]
        )
    )

    # --------------------------------------------------------
    # Feature preparation
    # --------------------------------------------------------

    precompute_all_features(
        trial_imgidx
    )

    all_features = np.load(
        FEATURE_FILE,
        mmap_mode="r",
    )

    expected_shape = (
        len(trial_imgidx),
        512,
    )

    if all_features.shape != expected_shape:
        raise RuntimeError(
            f"feature shape={all_features.shape}, "
            f"expected={expected_shape}"
        )

    # --------------------------------------------------------
    # Fixed val/test trial arrays
    # --------------------------------------------------------

    val_trials = images_to_trial_indices(
        trial_imgidx,
        split["val_images"],
    )

    test_trials = images_to_trial_indices(
        trial_imgidx,
        split["test_images"],
    )

    val_features = np.asarray(
        all_features[val_trials]
    )

    test_features = np.asarray(
        all_features[test_trials]
    )

    val_imgids = trial_imgidx[
        val_trials
    ]

    test_imgids = trial_imgidx[
        test_trials
    ]

    print(
        f"validation trials={len(val_trials)}"
    )

    print(
        f"test trials={len(test_trials)}"
    )

    if len(val_trials) != VAL_IMAGES * 3:
        raise RuntimeError(
            "validation trial 数错误"
        )

    if len(test_trials) != TEST_IMAGES * 3:
        raise RuntimeError(
            "test trial 数错误"
        )

    if args.prepare_only:
        print(
            "prepare-only 完成，不开始训练。"
        )
        return

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    results = []

    for seed in args.seeds:

        print("\n" + "=" * 72)
        print(
            f"SEED {seed}"
        )
        print("=" * 72)

        # 注意：
        # 所有 seed 共用同一个 nested_train_order。
        # seed 只用于 projector init + group shuffle。
        for n_images in args.scales:

            if (
                n_images <= 0
                or n_images > TRAIN_POOL_IMAGES
            ):
                raise ValueError(
                    f"scale={n_images} 超出 1..{TRAIN_POOL_IMAGES}"
                )

            train_images = (
                nested_train_order[
                    :n_images
                ]
            )

            train_trials = (
                images_to_trial_indices(
                    trial_imgidx,
                    train_images,
                )
            )

            expected = (
                n_images * 3
            )

            if len(train_trials) != expected:
                raise RuntimeError(
                    f"N={n_images}: "
                    f"得到 {len(train_trials)} trials，"
                    f"预期 {expected}"
                )

            print("\n" + "=" * 72)
            print(
                f"N_images={n_images} | "
                f"N_trials={expected} | "
                f"seed={seed}"
            )
            print("=" * 72)

            train_features = np.asarray(
                all_features[
                    train_trials
                ]
            )

            train_imgids = trial_imgidx[
                train_trials
            ]

            model, best_val = (
                train_one_scale(
                    train_features=train_features,
                    train_imgids=train_imgids,
                    val_features=val_features,
                    val_imgids=val_imgids,
                    clip_n=clip_n,
                    id2pos=id2pos,
                    val_candidates=split[
                        "val_images"
                    ],
                    seed=seed,
                    epochs=args.epochs,
                )
            )

            # ------------------------------------------------
            # Final fixed test evaluation
            # ------------------------------------------------

            test_q = encode_features(
                model,
                test_features,
            )

            t1, t5, t10, one = (
                evaluate_100way(
                    test_q,
                    test_imgids,
                    clip_n,
                    id2pos,
                    split[
                        "test_images"
                    ],
                )
            )

            # ------------------------------------------------
            # Save checkpoint
            # ------------------------------------------------

            model_path = (
                MODEL_DIR
                / f"mindeye2_N{n_images}_seed{seed}.pth"
            )

            torch.save(
                {
                    "model": model.state_dict(),
                    "seed": int(seed),
                    "n_images": int(n_images),
                    "n_trials": int(expected),
                    "best_val_top1": float(
                        best_val
                    ),
                },
                model_path,
            )

            row = {
                "seed": int(seed),
                "n_images": int(n_images),
                "n_trials": int(expected),
                "best_val_top1": float(
                    best_val
                ),
                "test_top1": float(t1),
                "test_top5": float(t5),
                "test_top10": float(t10),
                "test_1way": float(one),
            }

            results.append(
                row
            )

            print(
                f"[FINAL] "
                f"N={n_images} "
                f"trials={expected} | "
                f"valTop1={best_val:.5f} | "
                f"testTop1={t1:.5f} | "
                f"testTop5={t5:.5f} | "
                f"testTop10={t10:.5f} | "
                f"1way={one:.5f}",
                flush=True,
            )

    # --------------------------------------------------------
    # Save raw CSV
    # --------------------------------------------------------

    fields = [
        "seed",
        "n_images",
        "n_trials",
        "best_val_top1",
        "test_top1",
        "test_top5",
        "test_top10",
        "test_1way",
    ]

    with open(
        RESULT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            results
        )

    # --------------------------------------------------------
    # Aggregate by N
    # --------------------------------------------------------

    grouped = {}

    for row in results:
        grouped.setdefault(
            row["n_images"],
            [],
        ).append(row)

    summary_rows = []

    print("\n" + "=" * 72)
    print("SCALING SUMMARY")
    print("=" * 72)

    for n_images in args.scales:

        rows_n = grouped[
            n_images
        ]

        summary = {
            "n_images": int(
                n_images
            ),
            "n_trials": int(
                n_images * 3
            ),
        }

        for metric in [
            "test_top1",
            "test_top5",
            "test_top10",
            "test_1way",
        ]:

            vals = np.asarray(
                [
                    r[metric]
                    for r in rows_n
                ],
                dtype=np.float64,
            )

            mean = float(
                vals.mean()
            )

            std = float(
                vals.std(
                    ddof=0
                )
            )

            summary[
                metric + "_mean"
            ] = mean

            summary[
                metric + "_std"
            ] = std

            print(
                f"N={n_images:5d} "
                f"{metric:12s} "
                f"{mean:.5f} ± {std:.5f}"
            )

        summary_rows.append(
            summary
        )

    # --------------------------------------------------------
    # Power-law fit
    # --------------------------------------------------------

    fit_lines = []

    for metric in [
        "test_top1_mean",
        "test_top5_mean",
        "test_top10_mean",
        "test_1way_mean",
    ]:

        n = np.asarray(
            [
                r["n_images"]
                for r in summary_rows
            ],
            dtype=np.float64,
        )

        y = np.asarray(
            [
                r[metric]
                for r in summary_rows
            ],
            dtype=np.float64,
        )

        fit = fit_power_law(
            n,
            y,
        )

        if fit is None:
            line = (
                f"{metric}: fit failed"
            )
        else:
            line = (
                f"{metric}: "
                f"C={fit['C']:.6f}, "
                f"A={fit['A']:.6f}, "
                f"alpha={fit['alpha']:.6f}, "
                f"R2={fit['R2']:.6f}"
            )

        fit_lines.append(
            line
        )

        print(line)

    with open(
        FIT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "Probe Training-Data Scaling Law\n"
        )
        f.write(
            "Model: Accuracy(N) = C - A*N^(-alpha)\n\n"
        )

        for line in fit_lines:
            f.write(
                line + "\n"
            )

    print("\n" + "=" * 72)
    print(
        f"raw results: {RESULT_FILE}"
    )
    print(
        f"power-law fit: {FIT_FILE}"
    )
    print(
        f"models: {MODEL_DIR}"
    )
    print(
        f"features: {FEATURE_FILE}"
    )
    print(
        f"fixed split: {SPLIT_FILE}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
