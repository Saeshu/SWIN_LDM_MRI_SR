import torch


@torch.no_grad()
def ddim_step(eps_model, z, t, t_prev, const, cond=None):
    device = z.device
    B = z.shape[0]

    alpha_bar_t = const.alpha_bars[t].view(B, 1, 1, 1, 1)

    if isinstance(t_prev, int):
        alpha_bar_prev = const.alpha_bars[t_prev]
        alpha_bar_prev = torch.full_like(alpha_bar_t, alpha_bar_prev)
    else:
        alpha_bar_prev = const.alpha_bars[t_prev].view(B, 1, 1, 1, 1)

    # predict noise
    eps = eps_model(z, t, cond) if cond is not None else eps_model(z, t)

    # predict x0
    x0 = (z - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)

    # DDIM (eta = 0)
    z_prev = (
        torch.sqrt(alpha_bar_prev) * x0 +
        torch.sqrt(1 - alpha_bar_prev) * eps
    )

    return z_prev


@torch.no_grad()
def ddim_sample(
    eps_model,
    shape,
    schedule,
    const,
    cond=None,
    steps=50,
    t_min=200
):
    device = next(eps_model.parameters()).device

    z = torch.randn(shape, device=device)

    timesteps = torch.linspace(
        schedule.T - 1,
        t_min,
        steps,
        device=device
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
