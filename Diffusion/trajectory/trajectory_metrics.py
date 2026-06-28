"""
============================================================
trajectory_metrics.py

Trajectory metrics for latent diffusion.

Computes quantitative measures describing how the reverse
diffusion trajectory evolves.

No plotting.
No decoding.

Author:
    Diffusion Framework
============================================================
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F


# ----------------------------------------------------------
# Helper functions
# ----------------------------------------------------------

def _flatten(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(1)


def _safe_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.norm(x) + 1e-8


# ----------------------------------------------------------
# Distance to GT
# ----------------------------------------------------------

def distance_to_gt(
    x0_pred: torch.Tensor,
    residual_gt: torch.Tensor,
) -> float:
    """
    ||x0_pred - residual_gt||
    """

    return torch.norm(
        x0_pred - residual_gt
    ).item()


# ----------------------------------------------------------
# Step change
# ----------------------------------------------------------

def step_change(
    z: torch.Tensor,
    z_prev: Optional[torch.Tensor],
) -> float:

    if z_prev is None:
        return 0.0

    return torch.norm(
        z - z_prev
    ).item()


# ----------------------------------------------------------
# Contraction
# ----------------------------------------------------------

def contraction_ratio(
    z: torch.Tensor,
    z_prev: Optional[torch.Tensor],
) -> float:

    if z_prev is None:
        return 1.0

    return (
        _safe_norm(z)
        /
        _safe_norm(z_prev)
    ).item()


# ----------------------------------------------------------
# Direction cosine
# ----------------------------------------------------------

def direction_cosine(
    z: torch.Tensor,
    z_prev: Optional[torch.Tensor],
    residual_gt: torch.Tensor,
) -> float:
    """
    Compare

        update = z_prev - z

    against

        ideal = residual_gt - z_prev
    """

    if z_prev is None:
        return 1.0

    update = z_prev - z
    ideal = residual_gt - z_prev

    return F.cosine_similarity(
        _flatten(update),
        _flatten(ideal),
        dim=1
    ).mean().item()


# ----------------------------------------------------------
# Confidence
# ----------------------------------------------------------

def prediction_confidence(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """
    Relative prediction confidence.

    1.0 = perfect
    0.0 = random
    """

    err = torch.norm(
        prediction - target
    )

    denom = _safe_norm(target)

    return (1.0 - err / denom).item()


# ----------------------------------------------------------
# Main API
# ----------------------------------------------------------

def compute_trajectory_metrics(
    z: torch.Tensor,
    z_prev: Optional[torch.Tensor],
    x0_pred: torch.Tensor,
    residual_gt: torch.Tensor,
    v_pred: torch.Tensor,
    v_target: None,
) -> Dict[str, float]:

    if v_target is not None:
      metrics["confidence"] = prediction_confidence(
          v_pred,
          v_target,
      )
    else:
      metrics["confidence"] = float("nan")

    metrics = {

        "distance_to_gt":
            distance_to_gt(
                x0_pred,
                residual_gt,
            ),

        "step_change":
            step_change(
                z,
                z_prev,
            ),

        "contraction":
            contraction_ratio(
                z,
                z_prev,
            ),

        "direction_cosine":
            direction_cosine(
                z,
                z_prev,
                residual_gt,
            ),

        "confidence":
            prediction_confidence(
                v_pred,
                v_target,
            ),

    

}

    return metrics
def create_metric_history():

    return {
        "distance_to_gt": [],
        "step_change": [],
        "contraction": [],
        "direction_cosine": [],
        "confidence": [],
    }


def append_metrics(history, metrics):

    for k in history:
        history[k].append(metrics[k])

    return history
