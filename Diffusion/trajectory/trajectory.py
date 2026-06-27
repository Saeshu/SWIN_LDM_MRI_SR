"""
============================================================
trajectory.py

Runs one diffusion trajectory and records its evolution.

Author:
    Diffusion Framework
============================================================
"""

from typing import Dict

import torch

from Diffusion.losses import (
    predict_x0,
)

from Diffusion.trajectory_stats import (
    create_trajectory,
    update_trajectory,
    finalize_statistics,
)

from Diffusion.trajectory_metrics import (
    create_metric_history,
    append_metrics,
    compute_trajectory_metrics,
)

from Diffusion.trajectory_decode import (
    create_decoding_history,
    decode_trajectory_step,
    finalize_decoding_history,
)


# ----------------------------------------------------------
# Main API
# ----------------------------------------------------------

@torch.no_grad()
def run_trajectory(
    ae,
    unet,
    noise_scheduler,
    z_init,
    z_lr,
    residual_gt,
    v_target=None,          # <-- NEW
    w_e2=None,
    alpha=1.0,
    save_every=5,
):
    """
    Run a complete DDIM trajectory.

    Parameters
    ----------
    z_init
        Initial noisy residual latent.

    z_lr
        LR latent conditioning.

    residual_gt
        Ground-truth residual latent.

    Returns
    -------
    dict
    """

    device = z_init.device

    T = noise_scheduler.num_timesteps

    ##########################################################
    # Containers
    ##########################################################

    statistics = create_trajectory()

    metric_history = create_metric_history()

    decoding_history = create_decoding_history()

    ##########################################################
    # Initial latent
    ##########################################################

    z = z_init.clone()

    z_prev = None

    ##########################################################
    # DDIM Loop
    ##########################################################

    for t in reversed(range(T)):

        t_tensor = torch.full(
            (z.shape[0],),
            t,
            device=device,
            dtype=torch.long,
        )

        ######################################################
        # Prediction
        ######################################################

        v_pred = unet(

            z,

            t_tensor,

            cond=z_lr,

            w_e2=w_e2,

            alpha=alpha,

        )

        ######################################################
        # Recover x0
        ######################################################

        alpha_bar = noise_scheduler.alpha_bars[t]

        alpha_bar = alpha_bar.view(
            1,
            1,
            1,
            1,
            1,
        )

        x0_pred = predict_x0(

            z,

            v_pred,

            alpha_bar,

        )

        ######################################################
        # Recover epsilon
        ######################################################

        eps = (

            torch.sqrt(alpha_bar) * v_pred

            +

            torch.sqrt(1.0 - alpha_bar) * z

        )

        ######################################################
        # DDIM update
        ######################################################

        if t > 0:

            alpha_prev = noise_scheduler.alpha_bars[t - 1]

        else:

            alpha_prev = torch.tensor(
                1.0,
                device=device,
            )

        alpha_prev = alpha_prev.view(
            1,
            1,
            1,
            1,
            1,
        )

        z_new = (

            torch.sqrt(alpha_prev) * x0_pred

            +

            torch.sqrt(1.0 - alpha_prev) * eps

        )

        ######################################################
        # Statistics
        ######################################################

        # self target only for confidence statistics
        update_trajectory(
            statistics,
            z,
            z_prev,
            x0_pred,
            residual_gt,
            v_pred,
            v_target,
        )

        ######################################################
        # Metrics
        ######################################################
        metrics = compute_trajectory_metrics(
            z,
            z_prev,
            x0_pred,
            residual_gt,
            v_pred,
            v_target,
        ) 
        append_metrics(

            metric_history,

            metrics,

        )

        ######################################################
        # Decode
        ######################################################

        if (

            t % save_every == 0

            or

            t == 0

        ):

            decode_trajectory_step(

                ae,

                decoding_history,

                t,

                x0_pred,

                z_lr,

                w_e2,

            )

        ######################################################
        # Next step
        ######################################################

        z_prev = z.clone()

        z = z_new

    ##########################################################
    # Finalize
    ##########################################################

    statistics = finalize_statistics(
        statistics
    )

    decoding_history = finalize_decoding_history(
        decoding_history
    )

    return {

        "statistics": statistics,

        "metrics": metric_history,

        "decoding": decoding_history,

        "final_latent": z,

        "final_prediction": x0_pred,

    }
