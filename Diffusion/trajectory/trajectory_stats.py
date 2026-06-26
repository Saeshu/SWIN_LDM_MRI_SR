# Diffusion/trajectory_stats.py

import torch
import torch.nn.functional as F


# ============================================================
# Helpers
# ============================================================

def _flatten(x):
    return x.flatten(1)


def _safe_corr(x, y):
    """
    Pearson correlation between two tensors.
    """

    x = x.flatten()
    y = y.flatten()

    if x.std() < 1e-8 or y.std() < 1e-8:
        return 0.0

    return torch.corrcoef(torch.stack([x, y]))[0, 1].item()


def _cosine(x, y):
    return F.cosine_similarity(
        _flatten(x),
        _flatten(y),
        dim=1
    ).mean().item()


# ============================================================
# Empty trajectory container
# ============================================================

def create_trajectory():

    return {

        "latent": {

            "mean": [],
            "std": [],
            "norm": [],
            "energy": [],
            "min": [],
            "max": [],

        },

        "prediction": {

            "mse": [],
            "l1": [],
            "corr": [],
            "cosine": [],
            "confidence": [],

            "pred_mean": [],
            "pred_std": [],

            "target_mean": [],
            "target_std": [],

        },

        "residual": {

            "l1": [],
            "mse": [],
            "corr": [],
            "cosine": [],
            "norm_ratio": [],

        },

        "trajectory": {

            "distance_to_gt": [],
            "step_change": [],
            "contraction": [],
            "noise_std": [],

        }

    }


# ============================================================
# Latent statistics
# ============================================================

def latent_statistics(z):

    return {

        "mean": z.mean().item(),

        "std": z.std().item(),

        "norm": torch.norm(z).item(),

        "energy": torch.mean(z**2).item(),

        "min": z.min().item(),

        "max": z.max().item(),

    }


# ============================================================
# Prediction statistics
# ============================================================

def prediction_statistics(

    prediction,

    target,

):

    mse = F.mse_loss(
        prediction,
        target,
    ).item()

    l1 = F.l1_loss(
        prediction,
        target,
    ).item()

    cosine = _cosine(
        prediction,
        target,
    )

    corr = _safe_corr(
        prediction,
        target,
    )

    pred_norm = torch.norm(prediction)

    target_norm = torch.norm(target)

    confidence = 1.0 - (
        torch.norm(prediction - target)
        /
        (target_norm + 1e-8)
    ).item()

    return {

        "mse": mse,

        "l1": l1,

        "corr": corr,

        "cosine": cosine,

        "confidence": confidence,

        "pred_mean": prediction.mean().item(),

        "pred_std": prediction.std().item(),

        "target_mean": target.mean().item(),

        "target_std": target.std().item(),

    }


# ============================================================
# Residual statistics
# ============================================================

def residual_statistics(

    x0_pred,

    residual_gt,

):

    norm_ratio = (

        torch.norm(x0_pred)

        /

        (torch.norm(residual_gt) + 1e-8)

    ).item()

    return {

        "l1": F.l1_loss(
            x0_pred,
            residual_gt
        ).item(),

        "mse": F.mse_loss(
            x0_pred,
            residual_gt
        ).item(),

        "corr": _safe_corr(
            x0_pred,
            residual_gt
        ),

        "cosine": _cosine(
            x0_pred,
            residual_gt
        ),

        "norm_ratio": norm_ratio,

    }


# ============================================================
# Trajectory statistics
# ============================================================

def trajectory_statistics(

    z,

    z_prev,

    x0_pred,

    residual_gt,

):

    distance = torch.norm(
        x0_pred - residual_gt
    ).item()

    if z_prev is None:

        step_change = 0.0

        contraction = 1.0

    else:

        step_change = torch.norm(
            z - z_prev
        ).item()

        contraction = (

            torch.norm(z)

            /

            (torch.norm(z_prev) + 1e-8)

        ).item()

    return {

        "distance_to_gt": distance,

        "step_change": step_change,

        "contraction": contraction,

        "noise_std": z.std().item(),

    }


# ============================================================
# Append statistics
# ============================================================

def append_statistics(

    trajectory,

    latent,

    prediction,

    residual,

    traj,

):

    for k, v in latent.items():
        trajectory["latent"][k].append(v)

    for k, v in prediction.items():
        trajectory["prediction"][k].append(v)

    for k, v in residual.items():
        trajectory["residual"][k].append(v)

    for k, v in traj.items():
        trajectory["trajectory"][k].append(v)


# ============================================================
# Finalize
# ============================================================

def finalize_statistics(

    trajectory,

):

    for group in trajectory:

        for key in trajectory[group]:

            trajectory[group][key] = torch.tensor(

                trajectory[group][key]

            )

    return trajectory


# ============================================================
# Convenience wrapper
# ============================================================

def update_trajectory(

    trajectory,

    z,

    z_prev,

    x0_pred,

    residual_gt,

    v_pred,

    v_target,

):

    latent = latent_statistics(z)

    pred = prediction_statistics(

        v_pred,

        v_target,

    )

    residual = residual_statistics(

        x0_pred,

        residual_gt,

    )

    traj = trajectory_statistics(

        z,

        z_prev,

        x0_pred,

        residual_gt,

    )

    append_statistics(

        trajectory,

        latent,

        pred,

        residual,

        traj,

    )

    return trajectory
