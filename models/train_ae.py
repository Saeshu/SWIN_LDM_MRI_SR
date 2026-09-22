# scripts/train_ae.py

import os
import random

import numpy as np
import torch

from torch.utils.data import DataLoader

from src.models import AutoEncoder
from src.training.train import train_model


# ============================================================
# Configuration
# ============================================================

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_EPOCHS = 100
BATCH_SIZE = 1
NUM_WORKERS = 4

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_NAME = "ae_best.pt"

CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    CHECKPOINT_NAME,
)


# ============================================================
# Expert configuration
# ============================================================

ENCODER_EXPERTS = [
    "low",
    "point",
    "spatial",
    "depth",
    "identity_like",
]

DECODER_EXPERTS = [
    "low",
    "high",
    "spatial",
    "point",
    "depth",
]


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(SEED)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    print(
        f"Using device: {DEVICE}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    #
    # Replace these with your actual dataset construction.
    #
    # Keeping this section here makes the script the single
    # entry point for the complete training experiment.
    # --------------------------------------------------------

    train_dataset = ...
    val_dataset = ...

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = AutoEncoder(
        encoder_experts=ENCODER_EXPERTS,
        decoder_experts=DECODER_EXPERTS,
    )

    model = model.to(DEVICE)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Checkpoint directory
    # --------------------------------------------------------

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Print experiment configuration
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("Autoencoder Training")
    print("=" * 60)

    print(
        f"Seed:              {SEED}"
    )

    print(
        f"Device:            {DEVICE}"
    )

    print(
        f"Epochs:            {NUM_EPOCHS}"
    )

    print(
        f"Batch size:        {BATCH_SIZE}"
    )

    print(
        f"Learning rate:     {LEARNING_RATE}"
    )

    print(
        f"Weight decay:      {WEIGHT_DECAY}"
    )

    print(
        f"Encoder experts:   {ENCODER_EXPERTS}"
    )

    print(
        f"Decoder experts:   {DECODER_EXPERTS}"
    )

    print(
        f"Checkpoint:        {CHECKPOINT_PATH}"
    )

    print("=" * 60 + "\n")

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        save_path=CHECKPOINT_PATH,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\nTraining complete.")

    print(
        f"Final train loss: "
        f"{history['train_loss'][-1]:.6f}"
    )

    print(
        f"Final val loss:   "
        f"{history['val_loss'][-1]:.6f}"
    )

    print(
        f"Best checkpoint: "
        f"{CHECKPOINT_PATH}"
    )


if __name__ == "__main__":
    main()
