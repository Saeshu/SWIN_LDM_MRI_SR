"""
============================================================
trajectory_decode.py

Decode diffusion trajectory for visualization.

This module does NOT compute metrics or plots.
It simply converts latent predictions into image-space
representations and stores them for later visualization.

Author:
    Diffusion Framework

============================================================
"""

import torch
from typing import Dict, List


# ----------------------------------------------------------
# Slice extraction
# ----------------------------------------------------------

def extract_slices(
    volume: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Extract middle anatomical slices.

    Args
    ----
    volume : [B,C,D,H,W]

    Returns
    -------
    dict containing

        axial
        coronal
        sagittal
    """

    assert volume.ndim == 5

    volume = volume[0, 0]

    D, H, W = volume.shape

    return {

        "axial":
            volume[D // 2].cpu(),

        "coronal":
            volume[:, H // 2].cpu(),

        "sagittal":
            volume[:, :, W // 2].cpu(),

    }


# ----------------------------------------------------------
# Latent visualization
# ----------------------------------------------------------

def latent_snapshot(
    latent: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Convert latent into a visualizable form.

    Uses channel mean.

    Args
    ----
    latent : [B,C,D,H,W]
    """

    latent = latent.mean(dim=1, keepdim=True)

    return extract_slices(latent)


# ----------------------------------------------------------
# Decode one latent
# ----------------------------------------------------------

@torch.no_grad()
def decode_latent(
    ae,
    x0_pred: torch.Tensor,
    z_lr: torch.Tensor,
    w_e2=None,
):
    """
    Decode predicted residual.

    z_final = z_lr + x0_pred

    Returns
    -------
    dict
    """

    z_final = z_lr + x0_pred

    if w_e2 is None:
        recon = ae.decode(z_final)

    else:
        recon = ae.decode(z_final, w_e2)

    return {

    # visualization
    "latent": latent_snapshot(x0_pred),

    # 🔥 raw latent for downstream analyses
    "latent_raw": x0_pred.detach().cpu(),

    # decoded image
    "decoded": extract_slices(recon),

    # optional full reconstruction
    "full_volume": recon.detach().cpu(),

    }


# ----------------------------------------------------------
# Initialize trajectory
# ----------------------------------------------------------

def create_decoding_history():

    return {

        "timesteps": [],

        "latent": [],

        "decoded": [],

    }


# ----------------------------------------------------------
# Append one step
# ----------------------------------------------------------

def append_decoding_step(
    history,
    timestep,
    decoded_dict,
):

    history["timesteps"].append(int(timestep))

    history["latent"].append(
        decoded_dict["latent"]
    )

    history["decoded"].append(
        decoded_dict["decoded"]
    )


# ----------------------------------------------------------
# Finalize
# ----------------------------------------------------------

def finalize_decoding_history(
    history,
):

    return history


# ----------------------------------------------------------
# Main API
# ----------------------------------------------------------

@torch.no_grad()
def decode_trajectory_step(
    ae,
    history,
    timestep,
    x0_pred,
    z_lr,
    w_e2=None,
):
    """
    Decode one DDIM step.

    Parameters
    ----------
    history :
        trajectory dictionary

    timestep :
        current diffusion timestep

    x0_pred :
        predicted residual latent

    z_lr :
        LR latent

    w_e2 :
        routing maps
    """

    decoded = decode_latent(

        ae,

        x0_pred,

        z_lr,

        w_e2,

    )

    append_decoding_step(

        history,

        timestep,

        decoded,

    )

    return history
