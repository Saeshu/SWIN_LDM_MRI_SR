import torch

class LinearNoiseSchedule:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T

        betas = torch.linspace(beta_start, beta_end, T, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self

    def sample(self, z0, t):
        """
        z0: [B, C, D, H, W]
        t:  [B] (long)
        """
        noise = torch.randn_like(z0)

        alpha_bar = self.alpha_bars[t]                  # [B]
        alpha_bar = alpha_bar.view(-1, 1, 1, 1, 1)      # broadcast

        zt = torch.sqrt(alpha_bar) * z0 + \
             torch.sqrt(1 - alpha_bar) * noise

        return zt, noise


class NoiseScheduler:
    def __init__(self, num_timesteps=50, beta_max=0.008, schedule="cosine"):
        self.T = num_timesteps

        if schedule == "linear":
            self.betas = torch.linspace(1e-4, beta_max, self.T)
        elif schedule == "cosine":
            self.betas = self._cosine_schedule(beta_max)
        else:
            raise ValueError("Unknown schedule")

        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def _cosine_schedule(self, beta_max):
        t = torch.linspace(0, self.T, self.T + 1)
        f = torch.cos((t / self.T + 0.008) / 1.008 * math.pi / 2) ** 2
        alpha_bar = f / f[0]
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        return betas.clamp(max=beta_max)

    def add_noise(self, x0, t, noise):
        """
        x0: clean residual latent
        t: (B,)
        noise: N(0,1)
        """
        a_bar = self.alpha_bars[t].view(-1, 1, 1, 1, 1).to(x0.device)
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise


