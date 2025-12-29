import os
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from torch.utils.data import Dataset


class MRIDataset(Dataset):
    def __init__(
        self,
        root_dir,
        crop_size=None,
        normalize=True,
        downscale_factor=2
    ):
        self.root_dir = root_dir
        self.files = sorted([
            f for f in os.listdir(root_dir)
            if f.endswith(".nii") or f.endswith(".nii.gz")
        ])

        self.crop_size = crop_size
        self.normalize = normalize
        self.downscale_factor = downscale_factor

    def __len__(self):
        return len(self.files)

    def _random_crop(self, vol):
        D, H, W = vol.shape
        cd, ch, cw = self.crop_size

        sd = np.random.randint(0, D - cd + 1)
        sh = np.random.randint(0, H - ch + 1)
        sw = np.random.randint(0, W - cw + 1)

        return vol[
            sd:sd + cd,
            sh:sh + ch,
            sw:sw + cw
        ]

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, self.files[idx])

        nii = nib.load(path)
        vol = nii.get_fdata().astype(np.float32)

        # ensure 3D
        if vol.ndim == 4:
            vol = vol[..., 0]

        if self.crop_size is not None:
            vol = self._random_crop(vol)

        # normalization
        if self.normalize:
            vmin, vmax = np.percentile(vol, (1, 99))
            vol = np.clip(vol, vmin, vmax)
            vol = (vol - vmin) / (vmax - vmin + 1e-8)
            vol = vol * 2 - 1  # [-1, 1]

        vol = torch.from_numpy(vol).float()

        hr = vol.unsqueeze(0)

        lr = F.avg_pool3d(
            hr.unsqueeze(0),
            kernel_size=self.downscale_factor,
            stride=self.downscale_factor
        ).squeeze(0)

        return hr, lr 
