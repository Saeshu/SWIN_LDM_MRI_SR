"""
============================================================
losses.py

Core diffusion utilities and losses.

Everything in this file should be reusable and independent
of the AutoEncoder or any reconstruction losses.
============================================================
"""

import torch
import torch.nn.functional as F


# ============================================================
# TIMESTEP SAMPLING
# ============================================================

def sample_timesteps(
    batch_size,
    scheduler,
    device,
    bias="quadratic",
):
    """
    Sample diffusion timesteps.

    bias:
        "uniform"
        "quadratic" (default)
    """

    T = scheduler.num_timesteps

    if bias == "uniform":
        t = torch.randint(
            0,
            T,
            (batch_size,),
            device=device,
        )

    elif bias == "quadratic":

        u = torch.rand(
            batch_size,
            device=device,
        )

        t = ((u ** 2) * T).long()

    else:
        raise ValueError(
            f"Unknown bias {bias}"
        )

    return t.clamp(max=T - 1)


# ============================================================
# ALPHA BAR
# ============================================================

def get_alpha_bar(
    scheduler,
    t,
):
    """
    Returns alpha_bar_t
    shaped for broadcasting.
    """

    alpha_bar = scheduler.alpha_bars[t]

    return alpha_bar.view(
        -1,
        1,
        1,
        1,
        1,
    )


# ============================================================
# FORWARD DIFFUSION
# ============================================================

def add_noise(
    scheduler,
    z,
    t,
):
    """
    Forward diffusion.

    Returns
    -------
    z_t
    noise
    """

    noise = torch.randn_like(z)

    z_t = scheduler.add_noise(
        z,
        t,
        noise,
    )

    return z_t, noise


# ============================================================
# V TARGET
# ============================================================

def make_v_target(
    z,
    noise,
    alpha_bar,
):
    """
    Velocity parameterization.
    """

    return (

        torch.sqrt(alpha_bar) * noise

        -

        torch.sqrt(
            1.0 - alpha_bar
        ) * z

    )


# ============================================================
# X0 PREDICTION
# ============================================================

def predict_x0(
    z_t,
    v_pred,
    alpha_bar,
):
    """
    Recover x0 from v prediction.
    """

    x0 = (

        z_t

        -

        torch.sqrt(
            1 - alpha_bar
        ) * v_pred

    ) / torch.sqrt(
        alpha_bar
    )

    return x0


# ============================================================
# DIFFUSION LOSS
# ============================================================

def diffusion_loss(
    v_pred,
    v_target,
):
    """
    Standard MSE diffusion loss.
    """

    return F.mse_loss(
        v_pred,
        v_target,
    )


# ============================================================
# RESIDUAL LOSS
# ============================================================

def residual_loss(
    x0_pred,
    residual_gt,
):
    """
    L1 loss in residual latent space.
    """

    return F.l1_loss(
        x0_pred,
        residual_gt,
    )


# ============================================================
# RECONSTRUCTION LOSS
# ============================================================

def reconstruction_loss(
    x0_pred,
    z_lr,
    z_hr,
):
    """
    Compare reconstructed latent
    against HR latent.
    """

    z_final = z_lr + x0_pred

    return F.l1_loss(
        z_final,
        z_hr,
    )


# ============================================================
# TOTAL LOSS
# ============================================================

def total_loss(
    diffusion,
    residual=None,
    reconstruction=None,
    weights=None,
):
    """
    Combines losses.
    """

    if weights is None:

        weights = {

            "diffusion": 1.0,

            "residual": 0.5,

            "reconstruction": 0.5,

        }

    loss = weights["diffusion"] * diffusion

    if residual is not None:

        loss += (
            weights["residual"]
            * residual
        )

    if reconstruction is not None:

        loss += (
            weights["reconstruction"]
            * reconstruction
        )

    return loss


# ============================================================
# DEBUGGING
# ============================================================

@torch.no_grad()
def prediction_statistics(
    x0_pred,
    residual_gt,
):
    """
    Useful statistics for logging.
    """

    return {

        "pred_std":
            x0_pred.std().item(),

        "gt_std":
            residual_gt.std().item(),

        "pred_mean":
            x0_pred.mean().item(),

        "gt_mean":
            residual_gt.mean().item(),

        "l1":
            F.l1_loss(
                x0_pred,
                residual_gt,
            ).item(),

    }


# ============================================================
# NAN CHECK
# ============================================================

def check_finite(
    tensor,
    name="tensor",
):
    """
    Raises if NaN or Inf appears.
    """

    if not torch.isfinite(tensor).all():

        raise RuntimeError(
            f"{name} contains NaNs/Infs."
        )
