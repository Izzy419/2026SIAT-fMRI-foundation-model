# -*- coding: utf-8 -*-
"""
build_trial_tr.py  (方案2核心)
重建"每个试次 -> (session, run, 槽, 起始TR)" 的映射。
依据: 每run 75槽 x 4秒 = 300秒 = 188 TR x 1.6秒; stimpattern 标记哪些槽是真实试次。
用法:  python build_trial_tr.py
输出:  trial_to_tr.npy   (30000,) 每个试次的起始TR(全局帧索引)
"""
import scipy.io as sio
import numpy as np

mat = sio.loadmat('nsd_expdesign.mat', squeeze_me=True, struct_as_record=False)
stim = mat['stimpattern']          # (40, 12, 75): session, run, slot
TR = 1.6
SLOT_SEC = 4.0
TR_PER_SLOT = SLOT_SEC / TR        # 2.5 TR/槽

# ============ 1. 试次 -> (session, run, slot, run内起始TR) ============
trial_tr_inrun = []                # 每个试次在 run 内的起始 TR(按 run 内所有 TR 的全局计数)
trial_meta = []                    # (session, run, slot)
for s in range(40):
    counter = 0                    # 该 session 内真实试次计数
    for r in range(12):
        slots = np.where(stim[s, r] > 0)[0]      # 本 run 的真实试次槽
        for slot in slots:
            trial_tr_inrun.append(slot * TR_PER_SLOT)   # 槽起始 TR(浮点)
            trial_meta.append((s, r, int(slot)))
            counter += 1
    print(f'session {s+1}: {counter} 个真实试次')

trial_tr_inrun = np.array(trial_tr_inrun)
trial_meta = np.array(trial_meta)
print('总试次数:', len(trial_tr_inrun), ' (期望 30000)')

# ============ 2. 转成"全局 TR 帧索引"(跨 run 累计) ============
# 每个 run 固定 188 TR, 按 session+run 顺序累加
TR_PER_RUN = 188
run_start_tr = np.arange(40 * 12) * TR_PER_RUN   # 每个 run 的全局起始 TR
run_idx = trial_meta[:, 0] * 12 + trial_meta[:, 1]  # (session, run) -> run 全局编号
trial_tr_global = run_start_tr[run_idx] + np.round(trial_tr_inrun).astype(int)

print('前10个试次的映射:')
for t in range(10):
    s, r, slot = trial_meta[t]
    print(f'  试次{t}: session{s+1} run{r+1} 槽{slot}  run内TR={trial_tr_inrun[t]:.1f}  全局TR={trial_tr_global[t]}')

# ============ 3. 保存 ============
np.save('trial_to_tr.npy', trial_tr_global)
np.save('trial_meta.npy', trial_meta)
print('已保存 trial_to_tr.npy (30000,) 每个试次的全局起始TR帧')
