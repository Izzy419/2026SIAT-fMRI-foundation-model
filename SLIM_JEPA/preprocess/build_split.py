import h5py
import numpy as np

# ① 每个 betas 行的图片索引(NSD编号, 和 shared1000 同一套)
f = h5py.File('COCO_73k_subj_indices.hdf5', 'r')
imgidx = f['subj01'][:]        # (30000,)
f.close()
print('图片索引范围:', imgidx.min(), imgidx.max())

# ② 共享图布尔掩码
shared = np.load('shared1000.npy')   # (73000,) True=共享图
is_shared = shared[imgidx]           # 每个试次是不是共享图

# ③ 划分
train_idx = np.where(is_shared)[0]
test_idx  = np.where(~is_shared)[0]
print('训练试次(共享图):', len(train_idx))
print('测试试次(非共享图):', len(test_idx))

# ④ 保存
np.save('train_idx.npy', train_idx)
np.save('test_idx.npy', test_idx)
np.save('trial_imgidx.npy', imgidx)
print('已保存 train_idx / test_idx / trial_imgidx')
