"""
============================================================
logger.py

Training logger for diffusion models.

Responsible for

    - Tracking losses
    - Tracking validation metrics
    - Tracking trajectory metrics
    - Tracking MoE statistics
    - Saving history

No plotting.

Author:
    Diffusion Framework
============================================================
"""

from collections import defaultdict
from typing import Dict
import json
import torch


# ==========================================================
# Running Average
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

        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self):

        if self.count == 0:
            return 0.0

        return self.sum / self.count


# ==========================================================
# Main Logger
# ==========================================================

class TrainingLogger:

    def __init__(self):

        self.history = defaultdict(list)

        self.running = defaultdict(AverageMeter)

    # ------------------------------------------------------
    # Generic update
    # ------------------------------------------------------

    def update(
        self,
        **kwargs,
    ):

        for key, value in kwargs.items():

            if torch.is_tensor(value):

                value = value.item()

            self.running[key].update(value)

    # ------------------------------------------------------
    # End epoch
    # ------------------------------------------------------

    def end_epoch(self):

        epoch_stats = {}

        for key in self.running:

            avg = self.running[key].avg

            epoch_stats[key] = avg

            self.history[key].append(avg)

            self.running[key].reset()

        return epoch_stats

    # ------------------------------------------------------
    # Validation
    # ------------------------------------------------------

    def update_validation(
        self,
        metrics: Dict,
    ):

        for key, value in metrics.items():

            if torch.is_tensor(value):

                value = value.item()

            self.history[f"val_{key}"].append(value)

    # ------------------------------------------------------
    # Trajectory
    # ------------------------------------------------------

    def update_trajectory(
        self,
        trajectory: Dict,
    ):

        metrics = trajectory["metrics"]

        for key, value in metrics.items():

            if torch.is_tensor(value):

                value = value.mean().item()

            self.history[f"traj_{key}"].append(value)

    # ------------------------------------------------------
    # MoE
    # ------------------------------------------------------

    def update_moe(
        self,
        moe: Dict,
    ):

        entropy = moe["entropy"]

        gates = moe["gate_statistics"]

        for key, value in entropy.items():

            if torch.is_tensor(value):

                value = value.mean().item()

            self.history[f"moe_{key}"].append(value)

        for key, value in gates.items():

            if torch.is_tensor(value):

                value = value.mean().item()

            self.history[f"gate_{key}"].append(value)

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    def save_json(
        self,
        filename,
    ):

        history = {}

        for key, value in self.history.items():

            history[key] = [

                float(v)

                for v in value

            ]

        with open(filename, "w") as f:

            json.dump(

                history,

                f,

                indent=4,

            )

    # ------------------------------------------------------
    # Latest values
    # ------------------------------------------------------

    def latest(self):

        latest = {}

        for key in self.history:

            if len(self.history[key]):

                latest[key] = self.history[key][-1]

        return latest

    # ------------------------------------------------------
    # Print
    # ------------------------------------------------------

    def summary(self):

        latest = self.latest()

        print("=" * 60)

        for key, value in latest.items():

            print(f"{key:30s}: {value:.5f}")

        print("=" * 60)
