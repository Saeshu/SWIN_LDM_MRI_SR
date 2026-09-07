import os
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib

from torch.utils.data import Dataset


def center_crop_3d(vol, crop_size):
    D, H, W = vol.shape
    cd, ch, cw = crop_size

    assert D >= cd and H >= ch and W >= cw, (
        f"Volume {vol.shape} smaller than crop size {crop_size}"
    )

    d0 = (D - cd) // 2
    h0 = (H - ch) // 2
    w0 = (W - cw) // 2

    return vol[
        d0:d0 + cd,
        h0:h0 + ch,
        w0:w0 + cw,
    ]


def random_crop_3d(vol, crop_size):
    D, H, W = vol.shape
    cd, ch, cw = crop_size

    assert D >= cd and H >= ch and W >= cw, (
        f"Volume {vol.shape} smaller than crop size {crop_size}"
    )

    d = np.random.randint(0, D - cd + 1)
    h = np.random.randint(0, H - ch + 1)
    w = np.random.randint(0, W - cw + 1)

    return vol[
        d:d + cd,
        h:h + ch,
        w:w + cw,
    ]


def rician_noise(img, sigma):
    n1 = torch.randn_like(img) * sigma
    n2 = torch.randn_like(img) * sigma

    return torch.sqrt(
        (img + n1) ** 2 + n2 ** 2
    )


class MRIDataset(Dataset):
    """
    MRI dataset supporting:

    1. Autoencoder training:
        returns HR

    2. Super-resolution training:
        returns HR, LR_up, LR_small

    The dataset can either discover files from root_dir or
    use an explicitly provided list of files.
    """

    def __init__(
        self,
        root_dir=None,
        files=None,
        crop_size=(128, 128, 128),
        normalize=True,
        downscale_factor=None,
        noise_sigma=0.01,
        augment=True,
        random_crop=True,
    ):
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.normalize = normalize
        self.noise_sigma = noise_sigma
        self.augment = augment
        self.random_crop = random_crop

        # Normalize integer downscale factor into a 3D tuple
        if isinstance(downscale_factor, int):
            downscale_factor = (
                downscale_factor,
                downscale_factor,
                downscale_factor,
            )

        self.downscale_factor = downscale_factor

        # Use explicitly supplied split files if available
        if files is not None:
            self.files = sorted(files)

        # Otherwise discover all NIfTI files
        elif root_dir is not None:
            self.files = sorted([
                os.path.join(root_dir, filename)
                for filename in os.listdir(root_dir)
                if filename.endswith(".nii")
                or filename.endswith(".nii.gz")
            ])

        else:
            raise ValueError(
                "Either root_dir or files must be provided."
            )

        if len(self.files) == 0:
            raise RuntimeError("No NIfTI files found.")

        if self.downscale_factor is not None:
            for crop_dim, scale in zip(
                self.crop_size,
                self.downscale_factor,
            ):
                assert crop_dim % scale == 0, (
                    f"Crop dimension {crop_dim} must be divisible "
                    f"by scale factor {scale}."
                )

    def __len__(self):
        return len(self.files)

    def degrade(self, hr):
        """
        Create a low-resolution version and an upsampled LR version.

        Input:
            hr: [C, D, H, W]

        Returns:
            lr_up:    [C, D, H, W]
            lr_small: [C, d, h, w]
        """

        _, D, H, W = hr.shape

        sz, sy, sx = self.downscale_factor

        d = max(1, round(D / sz))
        h = max(1, round(H / sy))
        w = max(1, round(W / sx))

        lr_small = F.interpolate(
            hr.unsqueeze(0),
            size=(d, h, w),
            mode="trilinear",
            align_corners=False,
        )

        lr_up = F.interpolate(
            lr_small,
            size=(D, H, W),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)

        lr_small = lr_small.squeeze(0)

        # Add Rician noise to the upsampled LR image
        lr_up = rician_noise(
            lr_up,
            self.noise_sigma,
        )

        lr_up = lr_up.clamp(0, 1)

        return lr_up, lr_small

    def __getitem__(self, idx):
        filepath = self.files[idx]

        vol = nib.load(filepath).get_fdata().astype(
            np.float32
        )

        # Handle possible 4D NIfTI files
        if vol.ndim == 4:
            vol = vol[..., 0]

        # Convert from [H, W, D] to [D, H, W]
        vol = np.transpose(vol, (2, 0, 1))

        # Percentile-based intensity normalization
        if self.normalize:
            vmin, vmax = np.percentile(vol, (1, 99))

            vol = np.clip(
                vol,
                vmin,
                vmax,
            )

            vol = (
                vol - vmin
            ) / (
                vmax - vmin + 1e-8
            )

        # Crop the volume
        if self.random_crop:
            vol = random_crop_3d(
                vol,
                self.crop_size,
            )
        else:
            vol = center_crop_3d(
                vol,
                self.crop_size,
            )

        # Simple spatial augmentation
        if self.augment:
            if np.random.rand() < 0.5:
                vol = vol[:, :, ::-1]

            if np.random.rand() < 0.5:
                vol = vol[:, ::-1, :]

        # Convert to [C, D, H, W]
        hr = torch.from_numpy(
            vol.copy()
        ).float().unsqueeze(0)

        # Standard AE mode
        if self.downscale_factor is None:
            return hr

        # Super-resolution mode
        lr_up, lr_small = self.degrade(hr)

        return hr, lr_up, lr_small
