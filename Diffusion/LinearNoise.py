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
