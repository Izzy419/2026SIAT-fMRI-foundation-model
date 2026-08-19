#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
finetune_full_beta6000_probe_head_v2.py

目的：
验证“直接解冻 Slim，用 beta 训练”是否能通过 encoder adaptation
带来明显超过 Probe-6000 的提升，从而支持“架构具备分布迁移能力”的结论。

严格控制变量：
Probe-6000:
    beta -> Frozen Slim -> ClipProjector(512->1280)

Full-6000:
    beta -> Trainable Slim -> SAME ClipProjector(512->1280)

因此二者唯一核心差异：
    Slim encoder frozen vs trainable

数据：
    train = 6000 images * 3 = 18000 trials
    val   = 1000 images * 3 = 3000 trials
    test  = 3000 images * 3 = 9000 trials

split:
    probe_scaling_split_6000.npz

输入：
    NSDSingleTrial(beta, mask, trial_indices, t_frames=40)
    -> [B, 40, 96, 96, 96]
    -> unsqueeze(1)
    -> [B, 1, 40, 96, 96, 96]
    -> SlimEncoder.forward_features(...)

训练：
    multi-positive InfoNCE
    同图3个trial保持在同一个micro-batch

显存：
    2 images/micro-batch = 6 trials
    gradient accumulation = 4
"""

import os
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True"
)

import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.checkpoint import checkpoint

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

SLIM_CKPT = CHECKPOINT_PATH

TRIAL_IMGIDX_FILE = DATA / "trial_imgidx.npy"
CLIP_EMB_FILE = DATA / "clip_emb.npy"
CLIP_UIDX_FILE = DATA / "clip_unique_idx.npy"

SPLIT_FILE = DATA / "probe_scaling_split_6000.npz"

OUT_CKPT = DATA / \
    "slim_full_beta6000_probe_head_best_v2.pth"

OUT_RESULT = DATA / \
    "slim_full_beta6000_probe_head_result_v2.csv"


# ============================================================
# Experiment settings
# ============================================================

SEED = 0

TRAIN_IMAGES = 6000
VAL_IMAGES = 1000
TEST_IMAGES = 3000

EPOCHS = 30
VAL_EVERY = 2

# Full fine-tuning is memory sensitive. Each micro-batch keeps the three
# repetitions of two images together (6 trials). A FIFO target queue below
# restores a large negative set without inflating the 3-D encoder batch.
BATCH_IMAGES = 2
GRAD_ACCUM = 4

EVAL_BATCH = 4

LR_ENCODER = 1e-5
LR_HEAD = 1e-4
WEIGHT_DECAY = 0.05
TEMP = 0.1
MAX_GRAD_NORM = 1.0
NEG_QUEUE = 1024
USE_AMP = True
USE_ACTIVATION_CHECKPOINTING = True


# ============================================================
# Exact Slim config from working finetune_slim_v2.py
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
# Seed
# ============================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Same projector as Probe scaling
# ============================================================

class ClipProjector(nn.Module):

    def __init__(self, ind=512, outd=1280):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(ind, ind),
            nn.GELU(),
            nn.Linear(ind, ind),
            nn.GELU(),
            nn.Linear(ind, outd),
        )

    def forward(self, x):
        return F.normalize(
            self.mlp(x),
            dim=-1
        )


# ============================================================
# Beta dataset
# ============================================================

class PairedBetaDataset(Dataset):

    def __init__(
        self,
        trial_indices,
        clip_targets,
        image_ids,
    ):
        self.base = NSDSingleTrial(
            BETA,
            MASK_F,
            trial_indices,
            t_frames=40,
        )

        self.clip_targets = np.asarray(
            clip_targets,
            dtype=np.float32,
        )

        self.image_ids = np.asarray(
            image_ids,
            dtype=np.int64,
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):

        x = self.base[i]

        if isinstance(x, (tuple, list)):
            x = x[0]

        x = torch.as_tensor(
            x,
            dtype=torch.float32,
        )

        y = torch.from_numpy(
            self.clip_targets[i]
        ).float()

        image_id = int(
            self.image_ids[i]
        )

        return x, y, image_id


# ============================================================
# Group batch sampler
# Keep all 3 trials of each image together.
# ============================================================

class ImageGroupBatchSampler(Sampler):

    def __init__(
        self,
        image_ids,
        batch_images,
        seed,
        epoch,
    ):
        self.image_ids = np.asarray(
            image_ids,
            dtype=np.int64,
        )

        self.batch_images = int(
            batch_images
        )

        self.seed = int(seed)
        self.epoch = int(epoch)

        self.groups = []

        for image_id in np.unique(
            self.image_ids
        ):

            pos = np.flatnonzero(
                self.image_ids == image_id
            )

            if len(pos) != 3:
                raise RuntimeError(
                    f"image {int(image_id)} has "
                    f"{len(pos)} trials; expected 3"
                )

            self.groups.append(
                pos.tolist()
            )

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

            # only full image-groups
            if len(chosen) != self.batch_images:
                continue

            batch = []

            for g in chosen:
                batch.extend(
                    self.groups[int(g)]
                )

            yield batch

    def __len__(self):
        return (
            len(self.groups)
            // self.batch_images
        )


# ============================================================
# Foreground mask
# EXACT same logic as your working finetune_slim_v2.py
# ============================================================

@torch.no_grad()
def build_foreground_mask(
    first_trial_idx
):

    tmp = NSDSingleTrial(
        BETA,
        MASK_F,
        [int(first_trial_idx)],
        t_frames=40,
    )

    # NSDSingleTrial currently returns a Tensor [40,96,96,96], not a tuple.
    # A tuple is still accepted for backward compatibility with old loaders.
    item = tmp[0]
    vol = item[0] if isinstance(item, (tuple, list)) else item

    # Expected (40,96,96,96)
    vol = np.asarray(
        vol,
        dtype=np.float32,
    )

    if vol.ndim != 4:
        raise RuntimeError(
            f"Expected beta volume shape "
            f"(40,96,96,96), got {vol.shape}"
        )

    # Exactly as original:
    # take one 3-D volume for foreground mask.
    vol3d = vol[0]

    vt = (
        torch.from_numpy(vol3d)
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

    token_fore = (
        token_max > 0
    )

    mu = token_fore.view(
        1, 1,
        3, 8,
        3, 8,
        3, 8,
    ).amax(
        dim=(3, 5, 7)
    )

    mask = (
        mu.flatten()
        .bool()
        .repeat(5)
        .unsqueeze(0)
        .cuda()
    )

    if mask.shape != (1, 135):
        raise RuntimeError(
            f"Foreground mask shape should be "
            f"(1,135), got {tuple(mask.shape)}"
        )

    print(
        f"foreground mask: "
        f"{int(mask.sum())}/135",
        flush=True,
    )

    return mask


# ============================================================
# Full model
# ============================================================

class FullModel(nn.Module):

    def __init__(
        self,
        encoder,
        head,
        mask,
    ):
        super().__init__()

        self.encoder = encoder
        self.head = head

        self.register_buffer(
            "mask",
            mask,
            persistent=False,
        )

    def encoder_features(self, x, m):
        """SlimEncoder.forward_features with optional activation checkpointing.

        Checkpointing only recomputes transformer blocks during backward; it
        does not change the forward graph or the frozen/probe architecture.
        It is necessary here because the 96^3 fMRI encoder otherwise makes a
        six-trial full-finetuning batch unnecessarily fragile on a shared GPU.
        """
        pmask = m.view(x.shape[0], 1, *self.encoder.mask_spatial_shape)
        z = self.encoder.patch_embed(x, mask=pmask)
        z = z + self.encoder.get_pos_embed()
        z = self.encoder.unroll(z)
        visible = m[..., None].tile(1, self.encoder.mu_size, z.shape[2])
        z = z[visible].view(z.shape[0], -1, z.shape[-1])
        for block in self.encoder.blocks:
            if self.training and USE_ACTIVATION_CHECKPOINTING:
                z = checkpoint(block, z, use_reentrant=False)
            else:
                z = block(z)
        z = z.mean(dim=1)
        return self.encoder.encoder_norm(z)

    def forward(self, x):

        # CRITICAL:
        # NSDSingleTrial gives:
        #   [B,40,96,96,96]
        #
        # SlimEncoder expects:
        #   [B,1,40,96,96,96]
        #
        # This was the source of the previous AssertionError.
        if x.ndim != 5:
            raise RuntimeError(
                f"Expected beta tensor [B,40,96,96,96], "
                f"got {tuple(x.shape)}"
            )

        x = x.unsqueeze(1)

        if x.ndim != 6:
            raise RuntimeError(
                f"Expected Slim input to be 6D, "
                f"got {tuple(x.shape)}"
            )

        m = self.mask.repeat(
            x.shape[0],
            1,
        )

        z = self.encoder_features(x, m)

        return self.head(z)


# ============================================================
# Multi-positive InfoNCE
# ============================================================

def multipositive_nce(
    pred,
    target,
    image_ids,
):

    logits = (
        pred @ target.T
    ) / TEMP

    ids = image_ids.view(-1)

    positive_mask = (
        ids[:, None]
        == ids[None, :]
    )

    positive_logits = (
        logits.masked_fill(
            ~positive_mask,
            float("-inf"),
        )
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


def multipositive_nce_with_queue(
    pred,
    target,
    image_ids,
    queue_target,
    queue_ids,
):
    """Multi-positive InfoNCE with FIFO CLIP negatives.

    The micro-batch contains just two image groups to fit full SLIM tuning.
    Without this queue it has only one distinct negative image, unlike the
    frozen-probe experiment's 40-image batches. The queue changes *only* the
    negative set and never sends gradients through cached targets.
    """
    logits_batch = (pred @ target.T) / TEMP
    ids = image_ids.view(-1)
    positive = ids[:, None] == ids[None, :]
    log_positive = torch.logsumexp(
        logits_batch.masked_fill(~positive, float("-inf")), dim=1
    )
    if queue_target is None or len(queue_target) == 0:
        log_denom = torch.logsumexp(logits_batch, dim=1)
    else:
        logits_queue = (pred @ queue_target.T) / TEMP
        # A cached copy of the same image is another positive, not a negative.
        same_image = ids[:, None] == queue_ids[None, :]
        logits_queue = logits_queue.masked_fill(same_image, float("-inf"))
        log_denom = torch.logsumexp(
            torch.cat([logits_batch, logits_queue], dim=1), dim=1
        )
    return -(log_positive - log_denom).mean()


# ============================================================
# Batch encoding for validation/test
# ============================================================

@torch.no_grad()
def encode_dataset(
    model,
    dataset,
):

    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=EVAL_BATCH,
        shuffle=False,
        num_workers=0,
    )

    all_q = []
    all_ids = []

    for x, _, ids in loader:

        x = x.cuda(
            non_blocking=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=USE_AMP,
        ):
            q = model(x)

        all_q.append(
            q.float()
            .cpu()
            .numpy()
        )

        all_ids.append(
            ids.numpy()
        )

    return (
        np.concatenate(all_q),
        np.concatenate(all_ids),
    )


# ============================================================
# Exact image-level 100-way retrieval
# ============================================================

def evaluate_100way(
    q,
    image_ids,
    clip_n,
    id2pos,
    candidate_images,
):

    candidate_images = np.asarray(
        candidate_images,
        dtype=np.int64,
    )

    rng = np.random.default_rng(
        0
    )

    t1 = 0
    t5 = 0
    t10 = 0
    one = 0

    for i in range(
        len(q)
    ):

        gt = int(
            image_ids[i]
        )

        if gt not in candidate_images:
            raise RuntimeError(
                f"ground truth image {gt} "
                f"not in candidate images"
            )

        others = candidate_images[
            candidate_images != gt
        ]

        if len(others) < 99:
            raise RuntimeError(
                "Need >=99 distractors"
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
                id2pos[int(v)]
                for v in pool_ids
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

        t1 += gt_rank < 1
        t5 += gt_rank < 5
        t10 += gt_rank < 10

        d1 = int(
            rng.choice(
                others
            )
        )

        pair_pos = np.asarray(
            [
                id2pos[gt],
                id2pos[d1],
            ],
            dtype=np.int64,
        )

        one += int(
            np.argmax(
                clip_n[pair_pos]
                @ q[i]
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
# Main
# ============================================================

def main():

    seed_everything(
        SEED
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    device = torch.device(
        "cuda"
    )

    print(
        "=" * 72,
        flush=True,
    )

    print(
        "FULL FINETUNE 6000: "
        "beta -> Trainable Slim -> SAME Probe head -> CLIP",
        flush=True,
    )

    print(
        "=" * 72,
        flush=True,
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    trial_imgidx = np.load(
        TRIAL_IMGIDX_FILE
    ).astype(
        np.int64
    )

    clip_emb = np.load(
        CLIP_EMB_FILE
    ).astype(
        np.float32
    )

    clip_uidx = np.load(
        CLIP_UIDX_FILE
    ).astype(
        np.int64
    )

    clip_n = (
        clip_emb
        / (
            np.linalg.norm(
                clip_emb,
                axis=1,
                keepdims=True,
            )
            + 1e-9
        )
    ).astype(
        np.float32
    )

    id2pos = {
        int(image_id): int(pos)
        for pos, image_id
        in enumerate(clip_uidx)
    }

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    z = np.load(
        SPLIT_FILE
    )

    print(
        "split keys:",
        list(z.files),
        flush=True,
    )

    if "train_pool_images" in z:
        train_pool = np.asarray(
            z["train_pool_images"],
            dtype=np.int64,
        )
    elif "train_images" in z:
        train_pool = np.asarray(
            z["train_images"],
            dtype=np.int64,
        )
    else:
        raise KeyError(
            "No train image key in split."
        )

    if "val_images" in z:
        val_images = np.asarray(
            z["val_images"],
            dtype=np.int64,
        )
    elif "val_imgids" in z:
        val_images = np.asarray(
            z["val_imgids"],
            dtype=np.int64,
        )
    else:
        raise KeyError(
            "No validation image key in split."
        )

    if "test_images" in z:
        test_images = np.asarray(
            z["test_images"],
            dtype=np.int64,
        )
    elif "test_imgids" in z:
        test_images = np.asarray(
            z["test_imgids"],
            dtype=np.int64,
        )
    else:
        raise KeyError(
            "No test image key in split."
        )

    train_images = train_pool[
        :TRAIN_IMAGES
    ]

    train_set = set(
        int(x)
        for x in train_images
    )

    val_set = set(
        int(x)
        for x in val_images
    )

    test_set = set(
        int(x)
        for x in test_images
    )

    if train_set & val_set:
        raise RuntimeError(
            "train/val overlap"
        )

    if train_set & test_set:
        raise RuntimeError(
            "train/test overlap"
        )

    if val_set & test_set:
        raise RuntimeError(
            "val/test overlap"
        )

    # --------------------------------------------------------
    # Trial indices
    # --------------------------------------------------------

    train_trial_idx = np.flatnonzero(
        np.isin(
            trial_imgidx,
            train_images,
        )
    ).astype(
        np.int64
    )

    val_trial_idx = np.flatnonzero(
        np.isin(
            trial_imgidx,
            val_images,
        )
    ).astype(
        np.int64
    )

    test_trial_idx = np.flatnonzero(
        np.isin(
            trial_imgidx,
            test_images,
        )
    ).astype(
        np.int64
    )

    print(
        f"train: {len(train_images)} images / "
        f"{len(train_trial_idx)} trials",
        flush=True,
    )

    print(
        f"val:   {len(val_images)} images / "
        f"{len(val_trial_idx)} trials",
        flush=True,
    )

    print(
        f"test:  {len(test_images)} images / "
        f"{len(test_trial_idx)} trials",
        flush=True,
    )

    if len(train_trial_idx) != 18000:
        raise RuntimeError(
            "train is not 6000*3=18000 trials"
        )

    if len(val_trial_idx) != 3000:
        raise RuntimeError(
            "val is not 1000*3=3000 trials"
        )

    if len(test_trial_idx) != 9000:
        raise RuntimeError(
            "test is not 3000*3=9000 trials"
        )

    train_ids = trial_imgidx[
        train_trial_idx
    ]

    val_ids = trial_imgidx[
        val_trial_idx
    ]

    test_ids = trial_imgidx[
        test_trial_idx
    ]

    def clip_targets(ids):
        pos = np.asarray(
            [
                id2pos[int(v)]
                for v in ids
            ],
            dtype=np.int64,
        )
        return clip_n[pos]

    train_y = clip_targets(
        train_ids
    )

    val_y = clip_targets(
        val_ids
    )

    test_y = clip_targets(
        test_ids
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_ds = PairedBetaDataset(
        train_trial_idx,
        train_y,
        train_ids,
    )

    val_ds = PairedBetaDataset(
        val_trial_idx,
        val_y,
        val_ids,
    )

    test_ds = PairedBetaDataset(
        test_trial_idx,
        test_y,
        test_ids,
    )

    # --------------------------------------------------------
    # Mask
    # --------------------------------------------------------

    mask = build_foreground_mask(
        train_trial_idx[0]
    )

    # --------------------------------------------------------
    # Load Slim
    # --------------------------------------------------------

    print(
        "loading pretrained Slim...",
        flush=True,
    )

    encoder = SlimEncoder(
        **CFG
    )

    encoder.load_from_mae(
        SLIM_CKPT
    )

    # IMPORTANT:
    # Full finetuning: every encoder parameter trainable.
    for p in encoder.parameters():
        p.requires_grad = True

    head = ClipProjector()

    model = FullModel(
        encoder,
        head,
        mask,
    ).to(device)

    # --------------------------------------------------------
    # CRITICAL input-shape sanity check
    # --------------------------------------------------------

    model.eval()

    x0, _, _ = train_ds[0]

    print(
        f"dataset sample shape: {tuple(x0.shape)}",
        flush=True,
    )

    if tuple(x0.shape) != (
        40,
        96,
        96,
        96,
    ):
        raise RuntimeError(
            "Unexpected beta sample shape. "
            f"Got {tuple(x0.shape)}, expected "
            "(40,96,96,96)."
        )

    with torch.no_grad():

        x0b = x0.unsqueeze(
            0
        ).cuda()

        # This MUST become [1,1,40,96,96,96]
        print(
            f"Slim input shape: "
            f"{tuple(x0b.unsqueeze(1).shape)}",
            flush=True,
        )

        z0 = model.encoder_features(
            x0b.unsqueeze(1),
            mask,
        )

        print(
            f"Slim feature shape: "
            f"{tuple(z0.shape)}",
            flush=True,
        )

        if z0.ndim != 2 or z0.shape[1] != 512:
            raise RuntimeError(
                f"Expected [1,512] feature; "
                f"got {tuple(z0.shape)}"
            )

        del x0b, z0

    model.train()

    # --------------------------------------------------------
    # Trainable parameter count
    # --------------------------------------------------------

    enc_params = [
        p
        for p in model.encoder.parameters()
        if p.requires_grad
    ]

    head_params = [
        p
        for p in model.head.parameters()
        if p.requires_grad
    ]

    print(
        f"trainable encoder: "
        f"{sum(p.numel() for p in enc_params)/1e6:.2f}M",
        flush=True,
    )

    print(
        f"trainable head: "
        f"{sum(p.numel() for p in head_params)/1e6:.2f}M",
        flush=True,
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        [
            {
                "params": enc_params,
                "lr": LR_ENCODER,
            },
            {
                "params": head_params,
                "lr": LR_HEAD,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP,
    )

    # Cached CLIP targets keep full-tuning contrastive batches informative
    # despite the intentionally small 3-D fMRI micro-batch.
    queue_target = None
    queue_ids = None

    batches_per_epoch = (
        len(train_images)
        // BATCH_IMAGES
    )

    opt_steps = max(
        1,
        (
            EPOCHS
            * batches_per_epoch
        )
        // GRAD_ACCUM,
    )

    warmup_steps = max(
        1,
        int(
            0.05
            * opt_steps
        ),
    )

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(
                1,
                warmup_steps,
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
                    opt_steps
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

    best_val = -1.0
    best_epoch = -1

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    for epoch in range(
        EPOCHS
    ):

        t0 = time.time()

        model.train()

        sampler = ImageGroupBatchSampler(
            train_ids,
            BATCH_IMAGES,
            SEED,
            epoch,
        )

        loader = DataLoader(
            train_ds,
            batch_sampler=sampler,
            num_workers=0,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss_sum = 0.0
        micro_count = 0

        for step, (
            xb,
            yb,
            img_ids,
        ) in enumerate(
            loader
        ):

            xb = xb.cuda(
                non_blocking=True
            )

            yb = F.normalize(
                yb.cuda(
                    non_blocking=True
                ),
                dim=-1,
            )

            img_ids = img_ids.cuda(
                non_blocking=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=USE_AMP,
            ):
                pred = model(
                    xb
                )

                loss = multipositive_nce_with_queue(
                    pred.float(),
                    yb.float(),
                    img_ids,
                    queue_target,
                    queue_ids,
                )

            scaler.scale(
                loss / GRAD_ACCUM
            ).backward()

            loss_sum += float(
                loss.detach()
            )

            micro_count += 1

            if (
                (step + 1)
                % GRAD_ACCUM
                == 0
            ):

                scaler.unscale_(optimizer)

                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    MAX_GRAD_NORM,
                )

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

            # Enqueue detached, already-normalized targets after this loss;
            # the queue stores no gradient graph and is bounded in memory.
            with torch.no_grad():
                new_target = yb.detach()
                new_ids = img_ids.detach()
                if queue_target is None:
                    queue_target = new_target
                    queue_ids = new_ids
                else:
                    queue_target = torch.cat(
                        [queue_target, new_target], dim=0
                    )[-NEG_QUEUE:]
                    queue_ids = torch.cat(
                        [queue_ids, new_ids], dim=0
                    )[-NEG_QUEUE:]

        # Flush leftover gradients.
        if (
            micro_count % GRAD_ACCUM
            != 0
        ):

            scaler.unscale_(optimizer)

            nn.utils.clip_grad_norm_(
                model.parameters(),
                MAX_GRAD_NORM,
            )

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            optimizer.zero_grad(
                set_to_none=True
            )

        print(
            f"epoch {epoch+1}/{EPOCHS} "
            f"loss={loss_sum/max(1,micro_count):.5f} "
            f"time={time.time()-t0:.0f}s",
            flush=True,
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if (
            (epoch + 1)
            % VAL_EVERY
            == 0
            or epoch == EPOCHS - 1
        ):

            qv, iv = encode_dataset(
                model,
                val_ds,
            )

            v1, v5, v10, v1way = (
                evaluate_100way(
                    qv,
                    iv,
                    clip_n,
                    id2pos,
                    val_images,
                )
            )

            print(
                f"  VAL: "
                f"Top1={v1:.5f} "
                f"Top5={v5:.5f} "
                f"Top10={v10:.5f} "
                f"1way={v1way:.5f}",
                flush=True,
            )

            if v1 > best_val:

                best_val = float(v1)
                best_epoch = (
                    epoch + 1
                )

                torch.save(
                    {
                        "model": model.state_dict(),
                        "best_epoch": best_epoch,
                        "best_val_top1": best_val,
                        "train_images": TRAIN_IMAGES,
                        "train_trials": 18000,
                        "val_images": VAL_IMAGES,
                        "test_images": TEST_IMAGES,
                    },
                    OUT_CKPT,
                )

                print(
                    f"  [SAVE] {OUT_CKPT}",
                    flush=True,
                )

    # --------------------------------------------------------
    # Load best
    # --------------------------------------------------------

    ckpt = torch.load(
        OUT_CKPT,
        map_location=device,
    )

    model.load_state_dict(
        ckpt["model"]
    )

    # --------------------------------------------------------
    # Final test
    # --------------------------------------------------------

    qt, it = encode_dataset(
        model,
        test_ds,
    )

    t1, t5, t10, t1way = (
        evaluate_100way(
            qt,
            it,
            clip_n,
            id2pos,
            test_images,
        )
    )

    print(
        "=" * 72,
        flush=True,
    )

    print(
        f"FINAL TEST: "
        f"Top1={t1:.5f} "
        f"Top5={t5:.5f} "
        f"Top10={t10:.5f} "
        f"1way={t1way:.5f}",
        flush=True,
    )

    print(
        f"best epoch={best_epoch}, "
        f"best val Top1={best_val:.5f}",
        flush=True,
    )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    with open(
        OUT_RESULT,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "train_images",
                "train_trials",
                "val_images",
                "val_trials",
                "test_images",
                "test_trials",
                "best_epoch",
                "best_val_top1",
                "test_top1",
                "test_top5",
                "test_top10",
                "test_1way",
            ]
        )

        writer.writerow(
            [
                6000,
                18000,
                1000,
                3000,
                3000,
                9000,
                best_epoch,
                best_val,
                t1,
                t5,
                t10,
                t1way,
            ]
        )

    print(
        f"checkpoint: {OUT_CKPT}",
        flush=True,
    )

    print(
        f"result: {OUT_RESULT}",
        flush=True,
    )


if __name__ == "__main__":
    main()
