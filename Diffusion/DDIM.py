import torch


@torch.no_grad()
def ddim_step(eps_model, z, t, t_prev, const, cond=None):
    """
    One DDIM step (supports conditional or unconditional)

    z:       [B, C, D, H, W]
    t:       [B]
    t_prev:  scalar or [B]
    cond:    conditioning latent or None
    """

    device = z.device
    B = z.shape[0]

    # gather alpha bars safely
    alpha_bar_t = const.alpha_bars[t].view(B, 1, 1, 1, 1)

    if isinstance(t_prev, int):
        alpha_bar_prev = const.alpha_bars[t_prev]
        alpha_bar_prev = torch.full_like(alpha_bar_t, alpha_bar_prev)
    else:
        alpha_bar_prev = const.alpha_bars[t_prev].view(B, 1, 1, 1, 1)

    # predict noise
    if cond is None:
        eps = eps_model(z, t)
    else:
        eps = eps_model(z, t, cond)

    # predict x0
    x0 = (z - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)

    # DDIM update (eta = 0)
    z_prev = (
        torch.sqrt(alpha_bar_prev) * x0 +
        torch.sqrt(1.0 - alpha_bar_prev) * eps
    )

    return z_prev

@torch.no_grad()
def ddim_sample(eps_model, shape, schedule, const, cond=None, steps=50):
    device = next(eps_model.parameters()).device

    z = torch.randn(shape, device=device)

    timesteps = torch.linspace(
        schedule.T - 1, 0, steps, device=device
    ).long()

    for i in range(len(timesteps) - 1):
        t = torch.full((shape[0],), timesteps[i], device=device, dtype=torch.long)
        t_prev = timesteps[i + 1]

        z = ddim_step(
            eps_model=eps_model,
            z=z,
            t=t,
            t_prev=t_prev,
            const=const,
            cond=cond
        )

    return z

#sanity test
with torch.no_grad():
    z = ddim_sample(
        eps_model,
        shape=z_hr.shape,
        schedule=schedule,
        const=const,
        cond=z_cond,
        steps=20
    )