import torch

@torch.no_grad()
def sample_latent_ema(
    ema,
    noise_sched,
    cond,
    w_e2,
    device,
    guidance_scale=1.5,
    debug=False,
):

    ema.ema_model.eval()

    cond = cond.to(device)
    if w_e2 is not None:
        w_e2 = w_e2.to(device)

    z = torch.randn_like(cond)

    contractions = []

    T = noise_sched.num_timesteps

    x0_pred = None

    for t in reversed(range(T)):

        z_prev = z.clone()

        t_tensor = torch.full(
            (cond.shape[0],),
            t,
            device=device,
            dtype=torch.long,
        )

        ####################################################
        # Predict v
        ####################################################

        with torch.cuda.amp.autocast():

            if guidance_scale == 1.0:

                v_pred = ema.ema_model(
                    z=z,
                    t=t_tensor,
                    cond=cond,
                    w_e2=w_e2,
                    alpha=1.0,
                )

            else:

                v_uncond = ema.ema_model(
                    z=z,
                    t=t_tensor,
                    cond=None,
                    w_e2=None,
                    alpha=1.0,
                )

                v_cond = ema.ema_model(
                    z=z,
                    t=t_tensor,
                    cond=cond,
                    w_e2=w_e2,
                    alpha=1.0,
                )

                v_pred = v_uncond + guidance_scale * (v_cond - v_uncond)

            v_pred = torch.clamp(v_pred, -4, 4)

        ####################################################
        # Save x0 for visualization
        ####################################################

        alpha_bar = noise_sched.alpha_bars[t].to(device).view(
            1,1,1,1,1
        )

        x0_pred = (
            z
            -
            torch.sqrt(1-alpha_bar) * v_pred
        ) / torch.sqrt(alpha_bar)

        ####################################################
        # Scheduler handles reverse diffusion
        ####################################################

        z = noise_sched.step(
            z,
            t_tensor,
            v_pred,
        )

        ####################################################
        # Diagnostics
        ####################################################

        contraction = (
            torch.norm(z)
            /
            (torch.norm(z_prev)+1e-8)
        )

        contractions.append(contraction.item())

        if debug and t % 10 == 0:

            print(
                f"t={t:02d}"
                f" | latent std={z.std():.4f}"
                f" | x0 std={x0_pred.std():.4f}"
            )

    ########################################################
    # Residual -> SR latent
    ########################################################

    z_sr = cond + x0_pred

    ########################################################

    print(f"Mean |latent| : {z_sr.abs().mean():.4f}")
    print(f"Mean contraction : {sum(contractions)/len(contractions):.4f}")

    return {

        "latent": z_sr,

        "residual": x0_pred,

        "contractions": contractions,

    }
