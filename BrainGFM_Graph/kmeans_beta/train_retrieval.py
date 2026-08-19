# -*- coding: utf-8 -*-
"""
02_train_retrieval.py — BrainGFM + NSD 100-way retrieval training

本版相对上一版的关键修改：
1. 不加载 checkpoint，BrainGFM 从随机初始化开始训练。
2. 训练集内部按 IMAGE 划分 train/val：800/200 images，避免用 test 选 best。
3. 每张训练图的 3 个 trial 全部参与训练，不再每张图随机挑 1 个。
4. 使用 multi-positive InfoNCE：同一图片的 3 个 brain trial 对应同一 CLIP，
   互相视为 positive，不会再把同图 trial 当 negative。
5. 100-way validation：1 真图 + 99 个不重复干扰图。
6. 只有 validation Top-1 提升时才保存 best_retrieval.pth。
7. test 集只留给 03_eval_100way.py 做最终评估。

输入：01_build_dataset.py 的输出
输出：best_retrieval.pth / training_log.txt / split_info.npz
"""
import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from config import MODEL_DIR, PROCESSED_DIR

PROC = str(PROCESSED_DIR)
CODE_DIR = str(MODEL_DIR)

# ---------------- 超参数 ----------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
HIDDEN, CLIP_D = 256, 512

# BATCH 表示“每个 batch 的图片数”，每张图固定使用全部 3 trials。
BATCH_IMAGES = 16
EPOCHS = 150
LR = 1e-4
WARMUP = 10
TEMP = 0.07

VAL_IMAGES = 200                 # 1000 training images 中留 200 做 validation
TRAIN_IMAGES = 800
VAL_QUERY = 500
EVAL_EVERY = 1

FREEZE_ENCODER = False
SEED = 0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---------------- 数据 ----------------
print('加载数据 ...')
node_feat = np.load(os.path.join(HERE, 'node_features.npy'))   # [30000,100,1]
adj = np.load(os.path.join(HERE, 'adj.npy'))                   # [100,100]
trial_img = np.load(os.path.join(HERE, 'trial_img.npy')).astype(np.int64)
train_trials = np.load(os.path.join(HERE, 'train_trials.npy')).astype(np.int64)
test_trials = np.load(os.path.join(HERE, 'test_trials.npy')).astype(np.int64)
clip = np.load(os.path.join(PROC, 'clip_embeddings.npy')).astype(np.float32)

print('  node_feat:', node_feat.shape)
print('  adj:', adj.shape)
print('  train_trials:', len(train_trials))
print('  test_trials:', len(test_trials))
print('  clip:', clip.shape)

assert len(train_trials) == 3000, len(train_trials)
assert len(test_trials) == 27000, len(test_trials)

# ---------------- 按 IMAGE 划分 train / val ----------------
train_all_imgs = np.unique(trial_img[train_trials])
test_imgs = np.unique(trial_img[test_trials])

print('训练图数量:', len(train_all_imgs))
print('测试图数量:', len(test_imgs))
assert len(train_all_imgs) == 1000, len(train_all_imgs)
assert len(test_imgs) == 9000, len(test_imgs)

rng_split = np.random.default_rng(SEED)
shuffled_imgs = train_all_imgs.copy()
rng_split.shuffle(shuffled_imgs)

val_imgs = np.sort(shuffled_imgs[:VAL_IMAGES])
fit_imgs = np.sort(shuffled_imgs[VAL_IMAGES:])

assert len(fit_imgs) == TRAIN_IMAGES
assert len(val_imgs) == VAL_IMAGES
assert len(set(fit_imgs.tolist()) & set(val_imgs.tolist())) == 0
assert len(set(fit_imgs.tolist()) & set(test_imgs.tolist())) == 0
assert len(set(val_imgs.tolist()) & set(test_imgs.tolist())) == 0

fit_mask = np.isin(trial_img[train_trials], fit_imgs)
val_mask = np.isin(trial_img[train_trials], val_imgs)
fit_trials = train_trials[fit_mask]
val_trials = train_trials[val_mask]

print('TRAIN images/trials:', len(fit_imgs), len(fit_trials))
print('VAL   images/trials:', len(val_imgs), len(val_trials))
print('TEST  images/trials:', len(test_imgs), len(test_trials))

assert len(fit_trials) == TRAIN_IMAGES * 3
assert len(val_trials) == VAL_IMAGES * 3

np.savez(
    os.path.join(HERE, 'split_info.npz'),
    fit_imgs=fit_imgs,
    val_imgs=val_imgs,
    test_imgs=test_imgs,
    fit_trials=fit_trials,
    val_trials=val_trials,
)

# ---------------- 每张图的 trials ----------------
def group_trials_by_image(trials):
    groups = {}
    for t in trials.tolist():
        im = int(trial_img[t])
        groups.setdefault(im, []).append(int(t))

    for im, ts in groups.items():
        if len(ts) != 3:
            raise RuntimeError(
                '图片 %d 有 %d 个 trial，不是预期的 3 个。'
                % (im, len(ts))
            )
    return groups

fit_groups = group_trials_by_image(fit_trials)
val_groups = group_trials_by_image(val_trials)

# ---------------- 模型 ----------------
sys.path.insert(0, CODE_DIR)
from BrainGFM_Gprompt import BrainGFM

# 从零训练：不读取任何历史 checkpoint。
# 没有 pretrained.pth 时无法从 checkpoint 推断 MoE 配置，
# 因此固定使用单专家，与原始训练脚本的 moe_num_experts=1 一致。
ckpt = None
moe = 1
print('从零训练: 不加载 checkpoint; moe_num_experts = 1')



class RetModel(nn.Module):
    def __init__(self, moe):
        super().__init__()
        self.encoder = BrainGFM(
            hidden_dim=HIDDEN,
            ff_hidden_size=256,
            num_classes=2,
            num_self_att_layers=4,
            dropout=0.3,
            num_GNN_layers=4,
            nhead=8,
            max_feature_dim=512,
            rwse_steps=5,
            max_nodes=512,
            moe_num_experts=moe,
        )
        self.projector = nn.Sequential(
            nn.Linear(HIDDEN, 256),
            nn.ReLU(),
            nn.Linear(256, CLIP_D),
        )

    def forward(self, x, a, valid):
        g = self.encoder(
            x, a,
            parc_type='schaefer',
            disease_type='none',
            valid_num_nodes=valid
        )
        return self.projector(g)


model = RetModel(moe).to(DEVICE)

if FREEZE_ENCODER:
    for p in model.encoder.parameters():
        p.requires_grad = False
    print('encoder 已冻结，只训练 projector')

tot = sum(p.numel() for p in model.parameters())
trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('参数: 总=%d, 可训练=%d' % (tot, trn))

opt = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    betas=(0.5, 0.999)
)

# 固定 adjacency，避免每个 batch 重复 CPU->GPU 构造
adj_t = torch.from_numpy(adj).float().unsqueeze(0).to(DEVICE)

# ---------------- Multi-positive InfoNCE ----------------
def multi_positive_infonce(brain_emb, clip_emb, image_ids, temp):
    """
    brain_emb: [N,D]
    clip_emb:  [N,D]
    image_ids: [N]
    同 image 的样本全部视为 positive。
    """
    brain_emb = F.normalize(brain_emb, dim=-1)
    clip_emb = F.normalize(clip_emb, dim=-1)

    logits = brain_emb @ clip_emb.T / temp

    image_ids = image_ids.view(-1)
    pos_mask = image_ids[:, None].eq(image_ids[None, :])

    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    pos_count = pos_mask.sum(dim=1).clamp_min(1)
    loss = -(
        log_prob * pos_mask.float()
    ).sum(dim=1) / pos_count

    return loss.mean()


def make_train_batches():
    """
    每张图片一次性取它的全部 3 个 trials。
    BATCH_IMAGES=16 -> 一个 batch 通常 48 个 brain samples。
    """
    imgs = list(fit_groups.keys())
    random.shuffle(imgs)

    batches = []
    cur = []

    for im in imgs:
        cur.append(im)
        if len(cur) == BATCH_IMAGES:
            batches.append(cur)
            cur = []

    if cur:
        # 最后不足一个 batch 的图片也保留，不丢样本
        batches.append(cur)

    return batches


# ---------------- 100-way validation ----------------
@torch.no_grad()
def eval_100way(model, trials, candidate_imgs, n_query, seed):
    model.eval()

    rng = np.random.default_rng(seed)
    n_query = min(n_query, len(trials))
    qidx = rng.choice(len(trials), size=n_query, replace=False)
    qtrials = trials[qidx]

    embs = np.zeros((len(qtrials), CLIP_D), dtype=np.float32)

    for i0 in range(0, len(qtrials), 32):
        idx = qtrials[i0:i0 + 32]
        x = torch.from_numpy(node_feat[idx]).float().to(DEVICE)
        a = adj_t.expand(len(idx), -1, -1)
        e = model(x, a, [100] * len(idx))
        embs[i0:i0 + 32] = F.normalize(
            e, dim=-1
        ).cpu().numpy()

    clip_t = torch.from_numpy(clip).float()

    t1 = t5 = t10 = 0

    candidate_imgs = np.asarray(candidate_imgs, dtype=np.int64)

    for i in range(len(qtrials)):
        cid = int(trial_img[qtrials[i]])

        distractor_pool = candidate_imgs[candidate_imgs != cid]
        if len(distractor_pool) < 99:
            raise RuntimeError(
                'candidate pool 不足 99 个 distractors: %d'
                % len(distractor_pool)
            )

        distractors = rng.choice(
            distractor_pool,
            size=99,
            replace=False
        )
        cand = np.concatenate(
            [np.array([cid], dtype=np.int64), distractors]
        )

        ce = F.normalize(clip_t[cand], dim=-1)
        qe = torch.from_numpy(embs[i])

        sims = qe @ ce.T
        rank = int(torch.argsort(sims, descending=True).tolist().index(0)) + 1

        if rank <= 1:
            t1 += 1
        if rank <= 5:
            t5 += 1
        if rank <= 10:
            t10 += 1

    n = len(qtrials)
    return t1 / n * 100, t5 / n * 100, t10 / n * 100


# ---------------- 训练 ----------------
logf = os.path.join(HERE, 'training_log.txt')
# 每次重新训练时清空旧日志，避免混在一起
with open(logf, 'w') as f:
    f.write('Multi-positive InfoNCE training\n')
    f.write(
        'Train images=%d, Val images=%d, Test images=%d\n'
        % (len(fit_imgs), len(val_imgs), len(test_imgs))
    )

best_val_top1 = -1.0

print(
    '开始训练: %d epochs, %d images/batch (~%d trials/batch), '
    'lr=%.0e, temp=%.2f'
    % (EPOCHS, BATCH_IMAGES, BATCH_IMAGES * 3, LR, TEMP)
)

for ep in range(1, EPOCHS + 1):
    # warmup + cosine decay
    if ep <= WARMUP:
        lr = LR * ep / WARMUP
    else:
        p = (ep - WARMUP) / max(1, EPOCHS - WARMUP)
        lr = LR * 0.5 * (1 + np.cos(np.pi * p))

    for g in opt.param_groups:
        g['lr'] = lr

    model.train()
    batches = make_train_batches()

    tl = 0.0
    nb = 0
    t0 = time.time()

    for batch_imgs in batches:
        idx_list = []
        img_list = []

        for im in batch_imgs:
            for t in fit_groups[im]:
                idx_list.append(t)
                img_list.append(im)

        idx = np.asarray(idx_list, dtype=np.int64)
        img_ids = torch.from_numpy(
            np.asarray(img_list, dtype=np.int64)
        ).to(DEVICE)

        x = torch.from_numpy(node_feat[idx]).float().to(DEVICE)
        a = adj_t.expand(len(idx), -1, -1)

        # 每张图 3 个 trial 对应同一个 CLIP embedding
        clip_tgt = torch.from_numpy(
            clip[np.asarray(img_list, dtype=np.int64)]
        ).float().to(DEVICE)

        e = model(x, a, [100] * len(idx))

        loss = multi_positive_infonce(
            e, clip_tgt, img_ids, TEMP
        )

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        tl += loss.item()
        nb += 1

    avg_loss = tl / max(nb, 1)
    elapsed = time.time() - t0

    line = (
        'Epoch %3d/%d | loss=%.4f | lr=%.1e | %.1fs'
        % (ep, EPOCHS, avg_loss, lr, elapsed)
    )

    if ep % EVAL_EVERY == 0 or ep == 1:
        # 注意：这里只看 validation，不看 test
        v1, v5, v10 = eval_100way(
            model,
            val_trials,
            val_imgs,
            VAL_QUERY,
            seed=1000 + ep
        )

        line += (
            ' | VAL 100way T1/5/10: %.2f/%.2f/%.2f'
            % (v1, v5, v10)
        )

        if v1 > best_val_top1:
            best_val_top1 = v1
            torch.save(
                model.state_dict(),
                os.path.join(HERE, 'best_retrieval.pth')
            )
            line += '  <<< best val (saved)'

    print(line)
    with open(logf, 'a') as f:
        f.write(line + '\n')

print('✅ 训练完成。最佳 VAL 100-way Top-1 = %.2f%%' % best_val_top1)
print('   模型已保存为 best_retrieval.pth')
print('   最终 TEST 评估请运行: python 03_eval_100way.py')
