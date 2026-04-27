import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import timestep_embedding
from Diffusion.convsuite import TimeGatedConvSuite
#from Diffusion.schedule import SinusoidalTimeEmbedding
from Diffusion.LinearNoise import NoiseScheduler
#SinusoidalTimeEmbedding = SinusoidalTimeEmbedding(128)
scheduler = NoiseScheduler()

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: (B,) LongTensor
        """
        device = t.device
        half = self.dim // 2

        emb = torch.log(torch.tensor(10000.0, device=device)) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb

class TimeMLP(nn.Module):
    def __init__(self, tdim, channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(tdim, channels),
            nn.SiLU(),
            nn.Linear(channels, channels)
        )

    def forward(self, t_emb):
        return self.mlp(t_emb)[:, :, None, None, None]

class ResBlock3D(nn.Module):
    def __init__(self, channels, tdim):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.time_mlp = TimeMLP(tdim, channels)
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, x, t_emb):
        h = self.norm(x)
        h = F.silu(self.conv1(h))
        h = h + self.time_mlp(t_emb)
        h = F.silu(self.conv2(h))
        return x + h

class ConditionalEpsUNet3D(nn.Module):
    def __init__(
        self,
        z_ch,
        cond_ch,
        tdim=128,
        num_timesteps=50,
        use_temporal_suite=True,
    ):
        super().__init__()
        self.z_ch = z_ch
        self.cond_ch = cond_ch
        self.tdim = tdim
        self.num_timesteps = num_timesteps
        self.use_temporal_suite = use_temporal_suite

        self.time_embed = SinusoidalTimeEmbedding(self.tdim)

        # ✅ FIXED: input only takes z (no concat)
        self.in_conv = nn.Conv3d(z_ch, z_ch, 3, padding=1)

        # ---- encoder ----
        self.down = nn.Conv3d(z_ch, z_ch, 4, stride=2, padding=1)
        self.enc_block = ResBlock3D(z_ch, tdim)

        # ---- bottleneck ----
        self.mid_block = ResBlock3D(z_ch, tdim)

        if use_temporal_suite:
            self.temporal_suite = TimeGatedConvSuite(z_ch)
        else:
            self.temporal_suite = None

        # ---- decoder ----
        self.up = nn.ConvTranspose3d(z_ch, z_ch, 4, stride=2, padding=1)
        self.dec_block = ResBlock3D(z_ch, tdim)

        # ---- output ----
        self.out = nn.Conv3d(z_ch, z_ch, 3, padding=1)

    def forward(self, z, t, cond):
        """
        z:    (B, C, H, W, D)
        cond: (B, C, H, W, D)
        t:    (B,)
        """

        # ---- timestep embedding ----
        t_emb = timestep_embedding(t, self.tdim)

        # ✅ FIXED: no concat
        x = z
        x = self.in_conv(x)

        # ---- encoder ----
        x = self.down(x)
        x = self.enc_block(x, t_emb)

        # ---- bottleneck ----
        h = self.mid_block(x, t_emb)

        # ✅ CONDITIONING HERE (key part)
        if cond is not None:
            cond_mid = F.interpolate(
                cond,
                size=h.shape[2:],
                mode="trilinear",
                align_corners=False
            )
            h = h + cond_mid   # you can scale this later

        if self.temporal_suite is not None:
            h = h + self.temporal_suite(h, t)

        # ---- decoder ----
        h = self.up(h)
        h = self.dec_block(h, t_emb)

        return self.out(h)
