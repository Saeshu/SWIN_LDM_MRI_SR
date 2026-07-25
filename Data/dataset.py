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



def random_crop_3d(vol, crop_size):
    D, H, W = vol.shape
    cd, ch, cw = crop_size

    d = np.random.randint(0, D - cd + 1)
    h = np.random.randint(0, H - ch + 1)
    w = np.random.randint(0, W - cw + 1)

    return vol[d:d+cd, h:h+ch, w:w+cw]

def rician_noise(img, sigma):
    n1 = torch.randn_like(img) * sigma
    n2 = torch.randn_like(img) * sigma
    return torch.sqrt((img + n1) ** 2 + n2 ** 2)
  
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
        downscale_factor=None,   # None → AE mode, int → SR mode
        noise_sigma = 0.01,
        augment=True
    ):
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.normalize = normalize
        self.noise_sigma = noise_sigma
        if isinstance(downscale_factor, int):
            downscale_factor = (
                downscale_factor,
                downscale_factor,
                downscale_factor,
            )

        self.downscale_factor = downscale_factor
        self.augment = augment

        self.files = sorted([
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(".nii") or f.endswith(".nii.gz")
        ])

        if len(self.files) == 0:
            raise RuntimeError(f"No NIfTI files found in {root_dir}")

        # ensure divisibility for SR
        if downscale_factor is not None:
            for c, s in zip(crop_size, self.downscale_factor):
              assert c % s == 0
    
    def degrade(self, hr):
      """
      hr : [1, D, H, W]

      Returns
      -------
      lr_up   : [1,D,H,W] (input to model)
      lr_small: [1,d,h,w] (actual acquired image)
      """

      _, D, H, W = hr.shape

      sz, sy, sx = self.downscale_factor

      d = max(1, round(D / sz))
      h = max(1, round(H / sy))
      w = max(1, round(W / sx))

      # simulate acquisition
      lr_small = F.interpolate(
          hr.unsqueeze(0),
          size=(d, h, w),
          mode="trilinear",
          align_corners=False,
      )

      # bring back to HR grid
      lr_up = F.interpolate(
          lr_small,
          size=(D, H, W),
          mode="trilinear",
          align_corners=False,
      ).squeeze(0)

      # Rician noise
      lr_up = rician_noise(lr_up, self.noise_sigma)

      lr_up = lr_up.clamp(0, 1)

      return lr_up, lr_small.squeeze(0)
    
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # -----------------------------
        # Load volume
        # -----------------------------
        vol = nib.load(self.files[idx]).get_fdata().astype(np.float32)

        if vol.ndim == 4:
            vol = vol[..., 0]

        # [H, W, D] → [D, H, W]
        vol = np.transpose(vol, (2, 0, 1))

        # -----------------------------
        # Normalize (stable for AE)
        # -----------------------------
        if self.normalize:
            vmin, vmax = np.percentile(vol, (1, 99))
            vol = np.clip(vol, vmin, vmax)
            vol = (vol - vmin) / (vmax - vmin + 1e-8)  # 🔥 [0,1]

        # -----------------------------
        # Random crop (CRITICAL)
        # -----------------------------
        vol = random_crop_3d(vol, self.crop_size)

        # -----------------------------
        # Augmentation (optional)
        # -----------------------------
        if self.augment:
            if np.random.rand() < 0.5:
                vol = vol[:, :, ::-1]  # flip W

            if np.random.rand() < 0.5:
                vol = vol[:, ::-1, :]  # flip H

        # -----------------------------
        # Convert to tensor
        # -----------------------------
        hr = torch.from_numpy(vol.copy()).float().unsqueeze(0)  # [1, D, H, W]

        # -----------------------------
        # AE mode
        # -----------------------------
        if self.downscale_factor is None:
            return hr

        # -----------------------------
        # SR mode (spatial only)
        # -----------------------------
        lr, lr_small = self.degrade(hr)
 
        return hr, lr, lr_small
    
