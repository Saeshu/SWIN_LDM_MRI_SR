@torch.no_grad()
def sample_latent_ema(
    ema,
    noise_sched,
    shape,
    device,
    cond,                    # 🔥 z_lr (encoded LR latent, upsampled correctly)
    guidance_scale=1.5,
    debug=False
):
    ema.ema_model.eval()

    # -----------------------------
    # Initialize residual noise
    # -----------------------------
    z = torch.randn(shape, device=device)

    contractions = []
    T = noise_sched.num_timesteps

    for t in reversed(range(T)):
        z_prev = z.clone()

        t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)

        # -----------------------------
        # Predict v (CFG)
        # -----------------------------
        with torch.cuda.amp.autocast():

            if guidance_scale == 1.0:
                v_pred = ema.ema_model(z, t_tensor, cond=cond, alpha=1.0)

            else:
                v_uncond = ema.ema_model(z, t_tensor, cond=None, alpha=1.0)
                v_cond   = ema.ema_model(z, t_tensor, cond=cond, alpha=1.0)

                # 🔥 CFG (keep simple + stable)
                s = guidance_scale
                v_pred = v_uncond + s * (v_cond - v_uncond)

            v_pred = torch.clamp(v_pred, -4.0, 4.0)

        # -----------------------------
        # Get alpha_bar
        # -----------------------------
        alpha_bar_t = noise_sched.alpha_bars[t].to(device)
        alpha_bar_t = torch.clamp(alpha_bar_t, 1e-5, 1 - 1e-5)
        alpha_bar_t = alpha_bar_t.view(1, 1, 1, 1, 1)

        # -----------------------------
        # v → eps
        # -----------------------------
        eps = (
            torch.sqrt(alpha_bar_t) * v_pred +
            torch.sqrt(1 - alpha_bar_t) * z
        )

        # -----------------------------
        # x0 prediction (RESIDUAL)
        # -----------------------------
        x0_pred = (
            z - torch.sqrt(1 - alpha_bar_t) * eps
        ) / torch.sqrt(alpha_bar_t)

        # 🔥 stability clamp (important)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        # -----------------------------
        # previous timestep
        # -----------------------------
        if t > 0:
            alpha_bar_prev = noise_sched.alpha_bars[t - 1].to(device)
        else:
            alpha_bar_prev = torch.tensor(1.0, device=device)

        alpha_bar_prev = alpha_bar_prev.view(1, 1, 1, 1, 1)

        # -----------------------------
        # DDIM update (deterministic)
        # -----------------------------
        z = (
            torch.sqrt(alpha_bar_prev) * x0_pred +
            torch.sqrt(1 - alpha_bar_prev) * eps
        )

        # -----------------------------
        # Diagnostics
        # -----------------------------
        contraction = torch.norm(z) / (torch.norm(z_prev) + 1e-8)
        contractions.append(contraction.item())

        if debug and t % 10 == 0:
            print(
                f"t={t} | z std={z.std().item():.3f} | "
                f"x0 std={x0_pred.std().item():.3f}"
            )

        if not torch.isfinite(z).all():
            print("NaNs detected at t =", t)
            break

    # -----------------------------
    # 🔥 FINAL STEP (CRITICAL)
    # -----------------------------
    # z is residual → convert to full latent
    z_res_pred = x0_pred

    z_final = cond + z_res_pred   # 🔥 THIS IS THE KEY FIX

    # -----------------------------
    # Final stats
    # -----------------------------
    mean_abs = torch.mean(torch.abs(z_final)).item()
    print(f"mean |z_final|: {mean_abs:.4f}")
    print("max contraction:", max(contractions))
    print("mean contraction:", sum(contractions) / len(contractions))

    return z_final
