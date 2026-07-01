import torch


@torch.no_grad()
def sample_latent_ema(
    ema,
    noise_sched,
    cond,
    device,
    guidance_scale=1.5,
    debug=False,
):
    """
    Sample latent residual using EMA model.

    Parameters
    ----------
    ema
        EMA wrapper.

    noise_sched
        Noise scheduler.

    cond
        Encoded LR latent (already upsampled).

    Returns
    -------
    dict

        latent      : final SR latent

        residual    : predicted residual

        contractions: norm contraction curve
    """

    ema.ema_model.eval()

    ############################################################
    # Device
    ############################################################

    cond = cond.to(device)

    alpha_bars = noise_sched.alpha_bars.to(device)

    ############################################################
    # Initial residual noise
    ############################################################

    z = torch.randn_like(cond)

    contractions = []

    T = noise_sched.num_timesteps

    ############################################################
    # Reverse diffusion
    ############################################################

    for t in reversed(range(T)):

        z_prev = z.clone()

        t_tensor = torch.full(
            (cond.shape[0],),
            t,
            device=device,
            dtype=torch.long,
        )

        ########################################################
        # Predict v
        ########################################################

        with torch.amp.autocast("cuda"):

            if guidance_scale == 1.0:

                v_pred = ema.ema_model(

                    z,

                    t_tensor,

                    cond=cond,

                    alpha=1.0,

                )

            else:

                v_uncond = ema.ema_model(

                    z,

                    t_tensor,

                    cond=None,

                    alpha=1.0,

                )

                v_cond = ema.ema_model(

                    z,

                    t_tensor,

                    cond=cond,

                    alpha=1.0,

                )

                v_pred = (

                    v_uncond

                    +

                    guidance_scale

                    *

                    (v_cond - v_uncond)

                )

            v_pred = torch.clamp(

                v_pred,

                -4,

                4,

            )

        ########################################################
        # Alpha
        ########################################################

        alpha_bar_t = alpha_bars[t]

        alpha_bar_t = torch.clamp(

            alpha_bar_t,

            1e-5,

            1 - 1e-5,

        )

        alpha_bar_t = alpha_bar_t.view(

            1,

            1,

            1,

            1,

            1,

        )

        ########################################################
        # v -> eps
        ########################################################

        eps = (

            torch.sqrt(alpha_bar_t) * v_pred

            +

            torch.sqrt(1 - alpha_bar_t) * z

        )

        ########################################################
        # Predict residual
        ########################################################

        x0_pred = (

            z

            -

            torch.sqrt(1 - alpha_bar_t) * eps

        ) / torch.sqrt(alpha_bar_t)

        x0_pred = torch.clamp(

            x0_pred,

            -1,

            1,

        )

        ########################################################
        # Previous alpha
        ########################################################

        if t > 0:

            alpha_bar_prev = alpha_bars[t - 1]

        else:

            alpha_bar_prev = torch.tensor(

                1.0,

                device=device,

            )

        alpha_bar_prev = alpha_bar_prev.view(

            1,

            1,

            1,

            1,

            1,

        )

        ########################################################
        # DDIM update
        ########################################################

        z = (

            torch.sqrt(alpha_bar_prev) * x0_pred

            +

            torch.sqrt(1 - alpha_bar_prev) * eps

        )

        ########################################################
        # Diagnostics
        ########################################################

        contraction = (

            torch.norm(z)

            /

            (torch.norm(z_prev) + 1e-8)

        )

        contractions.append(

            contraction.item()

        )

        if debug and (t % 10 == 0):

            print(

                f"t={t:02d} | "

                f"z std={z.std():.4f} | "

                f"x0 std={x0_pred.std():.4f}"

            )

        if not torch.isfinite(z).all():

            print(

                f"NaNs detected at t={t}"

            )

            break

    ############################################################
    # Residual -> SR latent
    ############################################################

    z_res_pred = x0_pred

    z_sr = cond + z_res_pred

    ############################################################
    # Statistics
    ############################################################

    mean_abs = torch.mean(

        torch.abs(z_sr)

    ).item()

    print()

    print("=" * 60)

    print("Sampling statistics")

    print("=" * 60)

    print(f"|latent| mean : {mean_abs:.4f}")

    print(f"Max contraction : {max(contractions):.4f}")

    print(

        f"Mean contraction: "

        f"{sum(contractions)/len(contractions):.4f}"

    )

    print("=" * 60)

    ############################################################
    # Return
    ############################################################

    return {

        "latent": z_sr,

        "residual": z_res_pred,

        "contractions": contractions,

    }
