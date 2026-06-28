import torch
from torch.utils.data import DataLoader, random_split
from .dataset import MRIDataset


def create_dataloaders(
    data_root,
    batch_size=1,
    crop_size=(32, 128, 128),
    downscale_factor=None,
    num_workers=4,
    val_split=0.0,
    shuffle=True,
    drop_last=True,
    seed=42,
):

    dataset = MRIDataset(
        root_dir=data_root,
        crop_size=crop_size,
        downscale_factor=downscale_factor,
    )

    generator = torch.Generator().manual_seed(seed)

    ########################################################
    # No validation
    ########################################################

    if val_split <= 0:

        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=drop_last,
            generator=generator,
        )

        return train_loader, None

    ########################################################
    # Train / Validation split
    ########################################################

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=drop_last,
        generator=generator,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return train_loader, val_loader
