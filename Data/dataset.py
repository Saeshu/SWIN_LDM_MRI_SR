import os
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from torch.utils.data import Dataset

crop_size = (128, 256, 256)
def center_crop_3d(vol, crop_size):
    D, H, W = vol.shape
    cd, ch, cw = crop_size

    assert D >= cd and H >= ch and W >= cw, \
        f"Volume {vol.shape} smaller than crop {crop_size}"

    d0 = (D - cd) // 2
    h0 = (H - ch) // 2
    w0 = (W - cw) // 2

    return vol[
        d0:d0 + cd,
        h0:h0 + ch,
        w0:w0 + cw
    ]


class MRIDataset(Dataset):
    """
    MRI Dataset supporting:
    - Autoencoder training (HR only)
    - Super-resolution (HR + LR)
    """

    def __init__(
        self,
        root_dir,
        crop_size=(128, 256, 256),
        normalize=True,
        downscale_factor=None,  # None → AE mode
    ):
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.normalize = normalize
        self.downscale_factor = downscale_factor

        self.files = sorted([
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(".nii") or f.endswith(".nii.gz")
        ])

        if len(self.files) == 0:
            raise RuntimeError(f"No NIfTI files found in {root_dir}")

        # Safety: crop must be divisible by downscale
        if downscale_factor is not None:
            for c in crop_size:
                assert c % downscale_factor == 0, \
                    "crop_size must be divisible by downscale_factor"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        vol = nib.load(self.files[idx]).get_fdata().astype(np.float32)

        # ensure 3D
        if vol.ndim == 4:
            vol = vol[..., 0]

        # FIX AXIS ORDER
        vol = np.transpose(vol, (2,0,1))


        # normalize
        if self.normalize:
            vmin, vmax = np.percentile(vol, (1, 99))
            vol = np.clip(vol, vmin, vmax)
            vol = (vol - vmin) / (vmax - vmin + 1e-8)
            vol = vol * 2 - 1

        # center crop
        vol = center_crop_3d(vol, self.crop_size)

        hr = torch.from_numpy(vol).float().unsqueeze(0)
        assert hr.dtype == torch.float32
        # AE mode
        if self.downscale_factor is None:
            return hr

        # SR mode
        lr = F.avg_pool3d(
            hr.unsqueeze(0),
            kernel_size=self.downscale_factor,
            stride=self.downscale_factor
        ).squeeze(0)

        return hr, lr
