import torch
from torch.utils.data import DataLoader, random_split
from .dataset import MRIDataset


def create_dataloaders(
    data_root,
    batch_size=1,
    crop_size=(32, 128, 128),  # (D, H, W)
    downscale_factor=None,     # AE mode
    num_workers=4,
    val_split=0.1,
    shuffle=True,
):
    dataset = MRIDataset(
        root_dir=data_root,
        crop_size=crop_size,
        downscale_factor=downscale_factor,
    )

    if val_split > 0:
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size

        generator = torch.Generator().manual_seed(42)
        train_ds, val_ds = random_split(
            dataset, [train_size, val_size], generator=generator
        )
    else:
        train_ds = dataset
        val_ds = None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader
