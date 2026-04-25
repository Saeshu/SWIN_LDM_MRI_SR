import torch
import torch.nn as nn
import math 
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


class NoiseScheduler(nn.Module):
    def __init__(self, num_timesteps=50, schedule="linear"):
        super().__init__()
        self.num_timesteps = num_timesteps
    
        if schedule == "linear":
            # Strong low-frequency structure formation
            betas = torch.linspace(
                1e-4,      # beta_start
                2e-2,      # beta_end (IMPORTANT: larger than before)
                num_timesteps
            )
    
        elif schedule == "cosine":
            betas = self._cosine_schedule()
    
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
    
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
    
        # Register for safety if this is a module
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def _cosine_schedule(self, s=0.008):
        t = torch.linspace(0, self.num_timesteps, self.num_timesteps + 1)
        f = torch.cos(((t / self.num_timesteps) + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f / f[0]
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        return betas  # NO clamping

    def add_noise(self, x0, t, noise=None):
        # ---- HARD GUARDS ----
        assert x0.dim() == 5, f"x0 must be 5D (B,C,D,H,W), got {x0.shape}"
    
        if noise is None:
            noise = torch.randn_like(x0)
        else:
            assert noise.shape == x0.shape, "noise shape must match x0"
    
        # force timestep shape (B,)
        t = t.long().view(-1)
        assert t.dim() == 1
        assert t.shape[0] == x0.shape[0]
        assert t.max() < self.num_timesteps, "timestep out of range"
    
        # make sure buffers are on the right device
        alpha_bars = self.alpha_bars.to(x0.device)
    
        # gather alpha_bar_t
        a_bar = alpha_bars[t].view(-1, 1, 1, 1, 1)
    
        # DDPM forward process
        x_t = torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise
        return x_t

    def step(self, x_t, t, v_pred):
        """
        Reverse diffusion step using v-prediction
        x_t: (B,C,D,H,W)
        t:   (B,)
        v_pred: model output
        """
    
        t = t.long().view(-1)
    
        betas = self.betas.to(x_t.device)
        alphas = self.alphas.to(x_t.device)
        alpha_bars = self.alpha_bars.to(x_t.device)
    
        a_bar = alpha_bars[t].view(-1,1,1,1,1)
        a_bar_prev = alpha_bars[torch.clamp(t-1, min=0)].view(-1,1,1,1,1)
    
        # ---- recover x0 from v ----
        x0 = (
            torch.sqrt(a_bar) * x_t
            - torch.sqrt(1 - a_bar) * v_pred
        )
    
        # ---- compute eps from v ----
        eps = (
            torch.sqrt(1 - a_bar) * x_t
            + torch.sqrt(a_bar) * v_pred
        )
    
        # ---- DDPM mean ----
        mean = (
            torch.sqrt(a_bar_prev) * x0 +
            torch.sqrt(1 - a_bar_prev) * eps
        )
    
        # ---- add noise (except t=0) ----
        noise = torch.randn_like(x_t)
        nonzero_mask = (t > 0).float().view(-1,1,1,1,1)
    
        sigma = torch.sqrt(
            (1 - a_bar_prev) / (1 - a_bar) * betas[t].view(-1,1,1,1,1)
        )
    
        x_prev = mean + nonzero_mask * sigma * noise
    
        return x_prev




