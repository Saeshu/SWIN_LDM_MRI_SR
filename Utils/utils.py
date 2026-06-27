"""
============================================================
utils.py

General utilities for training.

Contains

    - Reproducibility
    - Device utilities
    - Parameter utilities
    - Gradient utilities
    - Model freezing
    - Running statistics

Author:
    Diffusion Framework
============================================================
"""

import os
import random
from typing import Any, Dict

import numpy as np
import torch


# ==========================================================
# Reproducibility
# ==========================================================

def seed_everything(seed: int = 42):
    """
    Seed Python, NumPy and PyTorch.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================================
# Device
# ==========================================================

def to_device(batch, device):
    """
    Move tensors, tuples, lists or dicts to device.
    """

    if torch.is_tensor(batch):
        return batch.to(device)

    if isinstance(batch, dict):
        return {
            k: to_device(v, device)
            for k, v in batch.items()
        }

    if isinstance(batch, (list, tuple)):
        return type(batch)(
            to_device(v, device)
            for v in batch
        )

    return batch


# ==========================================================
# Parameter counting
# ==========================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
    )


def count_trainable_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ==========================================================
# Gradient norm
# ==========================================================

def gradient_norm(model):

    total_norm = 0.0

    for p in model.parameters():

        if p.grad is None:
            continue

        param_norm = p.grad.detach().norm(2)

        total_norm += param_norm.item() ** 2

    return total_norm ** 0.5


# ==========================================================
# Freeze / unfreeze
# ==========================================================

def freeze(module):

    module.eval()

    for p in module.parameters():
        p.requires_grad = False


def unfreeze(module):

    module.train()

    for p in module.parameters():
        p.requires_grad = True


# ==========================================================
# Toggle requires_grad
# ==========================================================

def set_requires_grad(
    module,
    flag=True,
):

    for p in module.parameters():
        p.requires_grad = flag


# ==========================================================
# Detach dictionary
# ==========================================================

def detach_dict(d: Dict[str, Any]):

    out = {}

    for k, v in d.items():

        if torch.is_tensor(v):

            out[k] = v.detach()

        else:

            out[k] = v

    return out


# ==========================================================
# Average meter
# ==========================================================

class AverageMeter:

    def __init__(self):

        self.reset()

    def reset(self):

        self.sum = 0.0
        self.count = 0

    def update(
        self,
        value,
        n=1,
    ):

        self.sum += value * n
        self.count += n

    @property
    def avg(self):

        if self.count == 0:
            return 0.0

        return self.sum / self.count


# ==========================================================
# Running average
# ==========================================================

class RunningAverage:

    def __init__(self, momentum=0.95):

        self.momentum = momentum

        self.value = None

    def update(self, x):

        x = float(x)

        if self.value is None:

            self.value = x

        else:

            self.value = (

                self.momentum * self.value

                +

                (1 - self.momentum) * x

            )

        return self.value


# ==========================================================
# Parameter statistics
# ==========================================================

def parameter_statistics(model):

    stats = {}

    for name, p in model.named_parameters():

        stats[name] = {

            "mean": p.data.mean().item(),

            "std": p.data.std().item(),

            "min": p.data.min().item(),

            "max": p.data.max().item(),

        }

    return stats


# ==========================================================
# Model summary
# ==========================================================

def print_model_summary(model):

    total = count_parameters(model)
    trainable = count_trainable_parameters(model)

    print("=" * 60)
    print(type(model).__name__)
    print("=" * 60)
    print(f"Total Parameters     : {total:,}")
    print(f"Trainable Parameters : {trainable:,}")
    print("=" * 60)
