#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
冻结 SLIM（无 probe） + MindEye2 retrieval backbone：数据量缩放实验。

这不是把 SLIM 的 720 个中间 token 当作 MindEye2 的序列输入。那样会令
MindEye2 原始的 `backbone_linear(h * seq_len, 256 * 1664)` 膨胀为约 157B
参数，已经不是可运行、也不是可解释的实验。

本脚本使用已经提取好的 frozen SLIM 全局表征：

    beta -> frozen SLIM -> [512] -> MindEye2 BrainNetwork -> [256, 1664]
                             ^ 不加任何可训练 probe

其中 BrainNetwork、256x1664 OpenCLIP token 目标、MixCo/SoftCLIP 损失以及
token flatten 后的余弦检索均与 MindEye2 源码对应。与原版 MindEye2 的差异
仅是输入：原版由 beta 经线性 ridge 映射进入 h 维；这里由 frozen SLIM 的
512 维输出直接进入 h=512 的主干。因此此实验测量的是“SLIM 表征在强检索头
下的性能边界”，不应当和 MindEye2 的跨被试预训练结果直接等同。

固定划分：train pool 6000 images / val 1000 / test 3000；每图 3 trials。
默认运行 10 个规模 x 3 个初始化种子，支持断点续跑。

首次请先确认这些文件已经存在：
  probe_scaling_all_features.npy              (30000, 512)
  clip_tokens_256x1664_fp16.npy               (10000, 256, 1664)
  probe_scaling_split_6000.npz

建议先冒烟：
  CUDA_VISIBLE_DEVICES=0 python mindeye2_backbone_scaling.py --smoke-test
正式运行：
  CUDA_VISIBLE_DEVICES=0 nohup python mindeye2_backbone_scaling.py \
    > mindeye2_backbone_scaling.log 2>&1 &
"""

import argparse
import csv
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import sys
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from config import DATA_DIR

DATA = DATA_DIR
FEATURE_FILE = DATA / "probe_scaling_all_features.npy"
TOKEN_FILE = DATA / "clip_tokens_256x1664_fp16.npy"
TRIAL_FILE = DATA / "trial_imgidx.npy"
CLIP_UIDX_FILE = DATA / "clip_unique_idx.npy"
SPLIT_FILE = DATA / "probe_scaling_split_6000.npz"
# Keep prior failed/custom-head runs intact; this file is only for the
# faithful BrainNetwork implementation below.
RESULT_FILE = DATA / "mindeye2_backbone_scaling_results_v2.csv"
CKPT_DIR = DATA / "mindeye2_backbone_scaling_ckpts"

SCALES = [100, 200, 400, 800, 1200, 2000, 3000, 4000, 5000, 6000]
SEEDS = [0, 1, 2]

# MindEye2 uses four residual MLP-Mixer blocks and ViT-bigG patch tokens.
HIDDEN_DIM = 512          # frozen SLIM output; no trainable 512->h adapter
CLIP_SEQ_DIM = 256
CLIP_EMB_DIM = 1664
N_BLOCKS = 4
DROPOUT = 0.15

EPOCHS = 150
BATCH_SIZE = 24           # MindEye2 single-subject fine-tuning used 24
MAX_LR = 3e-4
WEIGHT_DECAY = 0.05
MIXUP_PCT = 0.33
MIXCO_TEMP = 0.006
SOFT_TEMP_START = 0.004
SOFT_TEMP_END = 0.0075
VAL_EVERY = 10
EVAL_BATCH = 4            # 4 * 100 token targets ~= 0.7 GB in fp32


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def amp_dtype(args):
    """bf16 has fp32-like exponent range; fp16 repeatedly overflowed here."""
    return torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16


class BrainNetwork(nn.Module):
    """MindEyeV2 src/models.py::BrainNetwork, with blurry branch disabled."""

    def __init__(self, h=HIDDEN_DIM, out_dim=CLIP_SEQ_DIM * CLIP_EMB_DIM,
                 seq_len=1, n_blocks=N_BLOCKS, drop=DROPOUT,
                 clip_size=CLIP_EMB_DIM):
        super().__init__()
        self.seq_len = seq_len
        self.h = h
        self.clip_size = clip_size
        self.mixer_blocks1 = nn.ModuleList(
            [self.mixer_block1(h, drop) for _ in range(n_blocks)]
        )
        self.mixer_blocks2 = nn.ModuleList(
            [self.mixer_block2(seq_len, drop) for _ in range(n_blocks)]
        )
        self.backbone_linear = nn.Linear(h * seq_len, out_dim, bias=True)
        self.clip_proj = self.projector(clip_size, clip_size, h=clip_size)

    @staticmethod
    def projector(in_dim, out_dim, h=2048):
        return nn.Sequential(
            nn.LayerNorm(in_dim), nn.GELU(), nn.Linear(in_dim, h),
            nn.LayerNorm(h), nn.GELU(), nn.Linear(h, h),
            nn.LayerNorm(h), nn.GELU(), nn.Linear(h, out_dim),
        )

    @staticmethod
    def mlp(in_dim, out_dim, drop):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.GELU(), nn.Dropout(drop),
            nn.Linear(out_dim, out_dim),
        )

    def mixer_block1(self, h, drop):
        return nn.Sequential(nn.LayerNorm(h), self.mlp(h, h, drop))

    def mixer_block2(self, seq_len, drop):
        return nn.Sequential(
            nn.LayerNorm(seq_len), self.mlp(seq_len, seq_len, drop)
        )

    def forward(self, x):
        # x: [B, 1, 512]. This exactly matches the original residual layout.
        residual1 = x
        residual2 = x.permute(0, 2, 1)
        for block1, block2 in zip(self.mixer_blocks1, self.mixer_blocks2):
            x = block1(x) + residual1
            residual1 = x
            x = x.permute(0, 2, 1)
            x = block2(x) + residual2
            residual2 = x
            x = x.permute(0, 2, 1)
        x = x.reshape(x.size(0), -1)
        backbone = self.backbone_linear(x).reshape(
            len(x), CLIP_SEQ_DIM, self.clip_size
        )
        retrieval = self.clip_proj(backbone)
        return backbone, retrieval


def mixco(voxels, beta=0.15, s_thresh=0.5):
    """MindEye2 utils.mixco, without in-place modification of the dataset tensor."""
    perm = torch.randperm(voxels.shape[0], device=voxels.device)
    betas = torch.distributions.Beta(beta, beta).sample([voxels.shape[0]]).to(
        voxels.device, dtype=voxels.dtype
    )
    select = (torch.rand(voxels.shape[0], device=voxels.device) <= s_thresh)
    shape = [-1] + [1] * (voxels.ndim - 1)
    mixed = voxels.clone()
    mixed[select] = (
        voxels[select] * betas[select].reshape(*shape)
        + voxels[perm][select] * (1 - betas[select]).reshape(*shape)
    )
    betas[~select] = 1
    return mixed, perm, betas, select


def mixco_nce(preds, targs, temp, perm, betas):
    """Bidirectional MindEye2 utils.mixco_nce."""
    logits = (preds @ targs.T) / temp
    probs = torch.diag(betas)
    probs[torch.arange(preds.shape[0], device=preds.device), perm] = 1 - betas
    loss_a = -(logits.log_softmax(-1) * probs).sum(-1).mean()
    loss_b = -(logits.T.log_softmax(-1) * probs.T).sum(-1).mean()
    return (loss_a + loss_b) / 2


def soft_clip_loss(preds, targs, temp):
    """MindEye2 utils.soft_clip_loss."""
    clip_clip = (targs @ targs.T) / temp
    brain_clip = (preds @ targs.T) / temp
    loss_a = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss_b = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    return (loss_a + loss_b) / 2


class TrialDataset(Dataset):
    def __init__(self, features, trial_positions):
        self.features = features
        self.trial_positions = np.asarray(trial_positions, dtype=np.int64)

    def __len__(self):
        return len(self.trial_positions)

    def __getitem__(self, i):
        return np.asarray(self.features[i], dtype=np.float32), self.trial_positions[i]


def load_metadata():
    for path in [FEATURE_FILE, TOKEN_FILE, TRIAL_FILE, CLIP_UIDX_FILE, SPLIT_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    features = np.load(FEATURE_FILE, mmap_mode="r")
    tokens = np.load(TOKEN_FILE, mmap_mode="r")
    trial_img = np.load(TRIAL_FILE).astype(np.int64)
    cidx = np.load(CLIP_UIDX_FILE).astype(np.int64)
    split = dict(np.load(SPLIT_FILE))

    if features.shape != (len(trial_img), HIDDEN_DIM):
        raise RuntimeError(
            f"Expected frozen SLIM features {(len(trial_img), HIDDEN_DIM)}, got {features.shape}"
        )
    if tokens.shape != (len(cidx), CLIP_SEQ_DIM, CLIP_EMB_DIM):
        raise RuntimeError(
            f"Expected token targets {(len(cidx), CLIP_SEQ_DIM, CLIP_EMB_DIM)}, got {tokens.shape}"
        )
    if not np.isfinite(np.asarray(features[:32], dtype=np.float32)).all():
        raise RuntimeError("Frozen SLIM features contain NaN/Inf.")

    id2pos = {int(image_id): pos for pos, image_id in enumerate(cidx)}
    try:
        trial_pos = np.asarray([id2pos[int(x)] for x in trial_img], dtype=np.int64)
        split_pos = {
            "train": np.asarray([id2pos[int(x)] for x in split["train_pool_images"]], dtype=np.int64),
            "val": np.asarray([id2pos[int(x)] for x in split["val_images"]], dtype=np.int64),
            "test": np.asarray([id2pos[int(x)] for x in split["test_images"]], dtype=np.int64),
        }
    except KeyError as exc:
        raise RuntimeError(f"Image id {exc.args[0]} is absent from clip_unique_idx.npy") from exc

    if len(split_pos["train"]) != 6000 or len(split_pos["val"]) != 1000 or len(split_pos["test"]) != 3000:
        raise RuntimeError("Split must be train=6000, val=1000, test=3000 images.")
    if (set(split_pos["train"]) & set(split_pos["val"]) or
            set(split_pos["train"]) & set(split_pos["test"]) or
            set(split_pos["val"]) & set(split_pos["test"])):
        raise RuntimeError("Image-level train/val/test split overlaps.")
    return features, tokens, trial_pos, split_pos


def make_trial_rows(trial_pos, image_positions):
    rows = np.flatnonzero(np.isin(trial_pos, image_positions)).astype(np.int64)
    expected = len(image_positions) * 3
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} trials, got {len(rows)}. Check split/trial alignment.")
    return rows


def cosine_temp(epoch, mix_epochs, total_epochs):
    soft_epochs = max(1, total_epochs - mix_epochs)
    pos = min(max(epoch - mix_epochs, 0), soft_epochs - 1)
    return SOFT_TEMP_END + (SOFT_TEMP_START - SOFT_TEMP_END) * 0.5 * (
        1 + math.cos(math.pi * pos / max(1, soft_epochs - 1))
    )


def make_pools(img_pos, candidate_positions, rng):
    pools = np.empty((len(img_pos), 100), dtype=np.int64)
    candidates = np.asarray(candidate_positions, dtype=np.int64)
    for i, gt in enumerate(img_pos):
        others = candidates[candidates != gt]
        pools[i, 0] = gt
        pools[i, 1:] = rng.choice(others, size=99, replace=False)
    return pools


@torch.no_grad()
def evaluate_100way(model, features, img_pos, tokens, candidate_positions, device):
    """MindEye2 token flatten cosine similarity, with fixed 100-way pools."""
    model.eval()
    rng = np.random.default_rng(20260813)
    pools = make_pools(img_pos, candidate_positions, rng)
    hits = np.zeros(3, dtype=np.int64)

    for start in range(0, len(img_pos), EVAL_BATCH):
        end = min(start + EVAL_BATCH, len(img_pos))
        xb = torch.from_numpy(np.asarray(features[start:end], dtype=np.float32)).to(device)
        _, q = model(xb.unsqueeze(1))
        q = F.normalize(q.flatten(1).float(), dim=-1)

        # float16 file is intentionally memory-mapped. Convert only one small 100-way pool batch.
        cand = np.asarray(tokens[pools[start:end]], dtype=np.float32)
        cand = torch.from_numpy(cand).to(device)
        cand = F.normalize(cand.flatten(2), dim=-1)
        scores = torch.einsum("bd,bkd->bk", q, cand)
        ranks = torch.argsort(scores, dim=1, descending=True)
        gt_rank = torch.argmax((ranks == 0).to(torch.int64), dim=1)
        hits += np.asarray([
            (gt_rank < 1).sum().item(),
            (gt_rank < 5).sum().item(),
            (gt_rank < 10).sum().item(),
        ])

    return tuple((hits / len(img_pos)).tolist())


def save_checkpoint(path, model, epoch, val_metrics, args):
    torch.save({
        "model": model.state_dict(),
        "epoch": epoch,
        "val_top1": val_metrics[0],
        "settings": vars(args),
    }, path)


def train_one_scale(n_images, seed, features, tokens, trial_pos, split_pos, args, device):
    set_seed(seed)
    train_images = split_pos["train"][:n_images]
    train_rows = make_trial_rows(trial_pos, train_images)
    val_rows = make_trial_rows(trial_pos, split_pos["val"])
    test_rows = make_trial_rows(trial_pos, split_pos["test"])

    ds = TrialDataset(features[train_rows], trial_pos[train_rows])
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=0, pin_memory=True)
    if len(loader) == 0:
        raise RuntimeError("Batch size is larger than the training set.")

    model = BrainNetwork().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  model parameters: {params / 1e6:.1f}M", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr,
                                  weight_decay=args.weight_decay)
    total_steps = args.epochs * len(loader)
    warm_steps = max(1, int(0.05 * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (step + 1) / warm_steps if step < warm_steps else
        0.5 * (1 + math.cos(math.pi * (step - warm_steps) /
                            max(1, total_steps - warm_steps))),
    )
    # GradScaler is only useful for fp16. BF16 avoids fp16's narrow exponent
    # range and is the stable default for this 256x1664-token objective.
    use_fp16_scaler = args.amp and args.amp_dtype == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)
    mix_epochs = int(args.epochs * args.mixup_pct)
    best_top1 = -1.0
    best_path = CKPT_DIR / f"mindeye2_frozen_slim_N{n_images}_seed{seed}.pth"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        for xb, pos in loader:
            xb = xb.to(device, non_blocking=True).unsqueeze(1)
            # CLIP target is accessed by CLIP position, never by COCO image id.
            yb = torch.from_numpy(np.asarray(tokens[pos.numpy()], dtype=np.float32)).to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype(args), enabled=args.amp):
                if epoch < mix_epochs:
                    xmix, perm, betas, _ = mixco(xb)
                    _, pred = model(xmix)
                    pred = F.normalize(pred.flatten(1).float(), dim=-1)
                    target = F.normalize(yb.flatten(1).float(), dim=-1)
                    loss = mixco_nce(pred, target, args.mixco_temp, perm, betas)
                else:
                    _, pred = model(xb)
                    pred = F.normalize(pred.flatten(1).float(), dim=-1)
                    target = F.normalize(yb.flatten(1).float(), dim=-1)
                    loss = soft_clip_loss(pred, target, cosine_temp(epoch, mix_epochs, args.epochs))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, scale {n_images}, seed {seed}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # GradScaler skips optimizer.step on fp16 overflow. Advancing the
            # LR schedule in that case consumes warm-up steps without learning.
            scale_before = scaler.get_scale() if use_fp16_scaler else None
            scaler.step(optimizer)
            scaler.update()
            if not use_fp16_scaler or scaler.get_scale() >= scale_before:
                scheduler.step()
            else:
                print("  AMP overflow: optimizer/scheduler step skipped", flush=True)
            total_loss += loss.detach().item()

        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            val = evaluate_100way(model, features[val_rows], trial_pos[val_rows], tokens,
                                  split_pos["val"], device)
            message = (
                f"  seed={seed} N={n_images} epoch={epoch + 1}/{args.epochs} "
                f"loss={total_loss / len(loader):.4f} val: "
                f"T1={val[0]:.4f} T5={val[1]:.4f} T10={val[2]:.4f}"
            )
            # This is not a reported metric. It separates two failure modes:
            # train-random -> optimization/implementation failure;
            # train-high + val-random -> severe low-data generalization limit.
            if args.diagnose_train:
                train_diag = evaluate_100way(
                    model, features[train_rows], trial_pos[train_rows], tokens,
                    train_images, device
                )
                message += (
                    f" | TRAIN-DIAG: T1={train_diag[0]:.4f} "
                    f"T5={train_diag[1]:.4f} T10={train_diag[2]:.4f}"
                )
            print(message + f" ({time.time() - t0:.0f}s)", flush=True)
            if val[0] > best_top1:
                best_top1 = val[0]
                save_checkpoint(best_path, model, epoch + 1, val, args)

    state = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    test = evaluate_100way(model, features[test_rows], trial_pos[test_rows], tokens,
                           split_pos["test"], device)
    del model, optimizer
    torch.cuda.empty_cache()
    return state["epoch"], state["val_top1"], test


def read_completed_rows(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(rows):
    fields = ["seed", "images", "trials", "best_epoch", "val_top1", "test_top1", "test_top5", "test_top10"]
    with open(RESULT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def smoke_test(features, tokens, trial_pos, split_pos, args, device):
    rows = make_trial_rows(trial_pos, split_pos["train"][:100])
    model = BrainNetwork().to(device).eval()
    x = torch.from_numpy(np.asarray(features[rows[:2]], dtype=np.float32)).to(device).unsqueeze(1)
    with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype(args), enabled=args.amp):
        backbone, retrieval = model(x)
        q = F.normalize(retrieval.flatten(1).float(), dim=-1)
    target = torch.from_numpy(np.asarray(tokens[trial_pos[rows[:2]]], dtype=np.float32)).to(device)
    target = F.normalize(target.flatten(1), dim=-1)
    loss = soft_clip_loss(q, target, SOFT_TEMP_END)
    if backbone.shape != (2, CLIP_SEQ_DIM, CLIP_EMB_DIM) or not torch.isfinite(loss):
        raise RuntimeError("MindEye2 smoke test failed.")
    print(f"SMOKE PASS: input=(2,1,512), backbone={tuple(backbone.shape)}, "
          f"retrieval={tuple(retrieval.shape)}, loss={loss.item():.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scales", nargs="+", type=int, default=SCALES)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-lr", type=float, default=MAX_LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--mixup-pct", type=float, default=MIXUP_PCT)
    parser.add_argument("--mixco-temp", type=float, default=MIXCO_TEMP)
    parser.add_argument("--val-every", type=int, default=VAL_EVERY)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16",
                        help="bf16 is the stable default; fp16 is retained only for comparison")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--diagnose-train", action="store_true",
                        help="also print in-sample 100-way retrieval; diagnostic only, never checkpoint selection")
    parser.add_argument("--overwrite-results", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if any(n not in SCALES for n in args.scales):
        raise ValueError(f"Scales must be a subset of {SCALES}")
    if args.batch_size < 2:
        raise ValueError("batch-size must be >= 2 for contrastive learning")

    device = torch.device("cuda")
    features, tokens, trial_pos, split_pos = load_metadata()
    print(f"frozen SLIM: {features.shape}; OpenCLIP tokens: {tokens.shape}", flush=True)
    print("split: train_pool=6000 images, val=1000 images, test=3000 images", flush=True)
    if args.smoke_test:
        smoke_test(features, tokens, trial_pos, split_pos, args, device)
        return

    old_rows = [] if args.overwrite_results else read_completed_rows(RESULT_FILE)
    completed = {(int(r["seed"]), int(r["images"])) for r in old_rows}
    rows = old_rows[:]
    for seed in args.seeds:
        for n_images in args.scales:
            if (seed, n_images) in completed:
                print(f"skip completed: seed={seed}, N={n_images}", flush=True)
                continue
            print(f"=== MindEye2 frozen-SLIM: seed={seed}, images={n_images}, trials={n_images * 3} ===", flush=True)
            best_epoch, val_top1, metrics = train_one_scale(
                n_images, seed, features, tokens, trial_pos, split_pos, args, device
            )
            row = {
                "seed": seed, "images": n_images, "trials": n_images * 3,
                "best_epoch": best_epoch, "val_top1": val_top1,
                "test_top1": metrics[0], "test_top5": metrics[1], "test_top10": metrics[2],
            }
            rows.append(row)
            write_rows(rows)
            print(f"FINAL seed={seed} N={n_images}: T1={metrics[0]:.4f} "
                  f"T5={metrics[1]:.4f} T10={metrics[2]:.4f}", flush=True)
    print(f"Done. Raw rows: {RESULT_FILE}", flush=True)


if __name__ == "__main__":
    main()
