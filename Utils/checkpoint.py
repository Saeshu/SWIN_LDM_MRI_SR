"""
============================================================
checkpoint.py

Checkpoint utilities for diffusion training.

Supports:

    - Save / Load
    - Resume training
    - EMA
    - AMP GradScaler
    - LR Scheduler
    - RNG State

Author:
    Diffusion Framework
============================================================
"""

import os
from pathlib import Path
from typing import Optional

import torch


# ==========================================================
# Save
# ==========================================================

def save_checkpoint(
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    ema=None,
    epoch: int = 0,
    path: str = "checkpoint.pt",
    extra: Optional[dict] = None,
):
    """
    Save complete training state.
    """

    checkpoint = {

        "epoch": epoch,

        "model": model.state_dict(),

    }

    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()

    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()

    if ema is not None:
        checkpoint["ema"] = ema.state_dict()

    # RNG states (important for exact resume)
    checkpoint["rng_state"] = torch.get_rng_state()

    if torch.cuda.is_available():
        checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state_all()

    if extra is not None:
        checkpoint["extra"] = extra

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, path)


# ==========================================================
# Load
# ==========================================================

def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    ema=None,
    map_location="cpu",
):
    """
    Load checkpoint.

    Returns
    -------
    epoch
    """

    ckpt = torch.load(
        path,
        map_location=map_location,
    )

    model.load_state_dict(
        ckpt["model"]
    )

    if optimizer is not None and "optimizer" in ckpt:

        optimizer.load_state_dict(
            ckpt["optimizer"]
        )

    if scheduler is not None and "scheduler" in ckpt:

        scheduler.load_state_dict(
            ckpt["scheduler"]
        )

    if scaler is not None and "scaler" in ckpt:

        scaler.load_state_dict(
            ckpt["scaler"]
        )

    if ema is not None and "ema" in ckpt:

        ema.load_state_dict(
            ckpt["ema"]
        )

    # Restore RNG

    if "rng_state" in ckpt:

        torch.set_rng_state(
            ckpt["rng_state"]
        )

    if (
        torch.cuda.is_available()
        and
        "cuda_rng_state" in ckpt
    ):

        torch.cuda.set_rng_state_all(
            ckpt["cuda_rng_state"]
        )

    epoch = ckpt.get("epoch", 0)

    extra = ckpt.get("extra", None)

    return epoch, extra


# ==========================================================
# Save Best
# ==========================================================

def save_best(
    model,
    metric,
    best_metric,
    optimizer=None,
    scheduler=None,
    scaler=None,
    ema=None,
    epoch=0,
    save_dir="./checkpoints",
):
    """
    Save only if metric improves.
    """

    improved = metric < best_metric

    if improved:

        save_checkpoint(

            model=model,

            optimizer=optimizer,

            scheduler=scheduler,

            scaler=scaler,

            ema=ema,

            epoch=epoch,

            path=os.path.join(
                save_dir,
                "best.pt",
            ),

            extra={
                "best_metric": metric
            }

        )

        best_metric = metric

    return best_metric


# ==========================================================
# Save Last
# ==========================================================

def save_last(
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    ema=None,
    epoch=0,
    save_dir="./checkpoints",
):

    save_checkpoint(

        model=model,

        optimizer=optimizer,

        scheduler=scheduler,

        scaler=scaler,

        ema=ema,

        epoch=epoch,

        path=os.path.join(
            save_dir,
            "last.pt",
        ),

    )


# ==========================================================
# Latest checkpoint
# ==========================================================

def latest_checkpoint(
    save_dir="./checkpoints",
):
    """
    Returns path to newest checkpoint.
    """

    save_dir = Path(save_dir)

    if not save_dir.exists():
        return None

    checkpoints = sorted(

        save_dir.glob("*.pt"),

        key=lambda p: p.stat().st_mtime,

    )

    if len(checkpoints) == 0:
        return None

    return str(checkpoints[-1])


# ==========================================================
# Check existence
# ==========================================================

def checkpoint_exists(
    path,
):
    return Path(path).exists()
