import h5py
import numpy as np
import torch
import nibabel as nib
from scipy.ndimage import affine_transform
from torch.utils.data import Dataset

class NSDSingleTrial(Dataset):
    def __init__(self, beta_file, mask_file, trial_indices, t_frames=40,
                 mean=None, std=None):
        with h5py.File(beta_file, 'r') as f:
            self.betas = f['betas'][:]
        mask = nib.load(mask_file)
        data = mask.get_fdata()
        self.coords = np.argwhere(data == 1).astype(int)
        self.grid_18 = data.shape
        self.aff = mask.affine
        self.matrix = np.diag([2/1.8]*3)
        self.offset = (-96 - self.aff[:3,3]) / 1.8
        self.t_frames = t_frames
        self.mean, self.std = mean, std
        self.trial_indices = np.asarray(trial_indices)

    def __len__(self):
        return len(self.trial_indices)

    def __getitem__(self, i):
        t = int(self.trial_indices[i])
        b = self.betas[t].astype(np.float32)
        if self.mean is not None:
            b = (b - self.mean) / (self.std + 1e-8)
        vol18 = np.zeros(self.grid_18, dtype=np.float32)
        c = self.coords
        vol18[c[:,0], c[:,1], c[:,2]] = b
        vol96 = affine_transform(vol18, self.matrix, offset=self.offset,
                                 output_shape=(96,96,96), order=1,
                                 mode='constant', cval=0)
        block = np.repeat(vol96[..., None], self.t_frames, axis=-1)
        return torch.from_numpy(block).permute(3,0,1,2)
