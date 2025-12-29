class LinearNoiseSchedule:
    def __init__(self, T=100, beta_start=1e-4, beta_end=0.02):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def sample(self, z0, t):
        noise = torch.randn_like(z0)
        alpha_bar = self.alpha_bars[t].to(z0.device)
        zt = torch.sqrt(alpha_bar) * z0 + torch.sqrt(1 - alpha_bar) * noise
        return zt, noise
