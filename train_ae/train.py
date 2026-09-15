import json

import torch
from torch.utils.data import DataLoader

from datasets.mri_dataset import MRIDataset
from models.your_model import YourModel
from training import train_model


def main():

    # --------------------------------------------------
    # Configuration
    # --------------------------------------------------

    root_dir = "/workspace/dataset"
    split_path = "/workspace/split/split_seed42.json"

    crop_size = (128, 128, 128)

    batch_size = 1
    num_workers = 4

    num_epochs = 100
    learning_rate = 1e-4

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # --------------------------------------------------
    # Load fixed split
    # --------------------------------------------------

    with open(split_path, "r") as f:
        split_data = json.load(f)

    train_files = split_data["train"]
    val_files = split_data["val"]

    # --------------------------------------------------
    # Create datasets
    # --------------------------------------------------

    train_dataset = MRIDataset(
        files=train_files,
        crop_size=crop_size,
        normalize=True,
        augment=True,
        random_crop=True,
    )

    val_dataset = MRIDataset(
        files=val_files,
        crop_size=crop_size,
        normalize=True,
        augment=False,
        random_crop=False,
    )

    # --------------------------------------------------
    # Create dataloaders
    # --------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))

    # --------------------------------------------------
    # Create model
    # --------------------------------------------------

    model = YourModel(
        # model arguments here
    ).to(device)

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    # --------------------------------------------------
    # Train
    # --------------------------------------------------

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        save_path="checkpoints/model_best.pt",
    )


if __name__ == "__main__":
    main()
