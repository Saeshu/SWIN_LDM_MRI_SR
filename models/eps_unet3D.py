import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import timestep_embedding
from Diffusion.convsuite import TimeGatedConvSuite
from Diffusion.LinearNoise import NoiseScheduler

scheduler = NoiseScheduler()


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
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
            nn.Linear(channels, channels),
        )

    def forward(self, t_emb):
        return self.mlp(t_emb)[:, :, None, None, None]


class ResBlock3D(nn.Module):
    """Now supports differing in/out channels (needed after concat skips)."""
    def __init__(self, in_ch, tdim, out_ch=None):
        super().__init__()
        out_ch = out_ch or in_ch
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = TimeMLP(tdim, out_ch)
        self.norm = nn.GroupNorm(8, in_ch)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.norm(x)
        h = F.silu(self.conv1(h))
        h = h + self.time_mlp(t_emb)
        h = F.silu(self.conv2(h))
        return self.skip(x) + h


class ConditionalEpsUNet3D(nn.Module):
    def __init__(self, z_ch, cond_ch, tdim=128, num_timesteps=50,
                 use_temporal_suite=True, normalize_input=True):
        super().__init__()
        self.z_ch = z_ch
        self.cond_ch = cond_ch
        self.tdim = tdim
        self.num_timesteps = num_timesteps
        self.use_temporal_suite = use_temporal_suite
        self.normalize_input = normalize_input

        self.time_embed = SinusoidalTimeEmbedding(self.tdim)

        # ---- input: FIX 2 concat conditioning ----
        self.in_conv = nn.Conv3d(z_ch + cond_ch, z_ch, 3, padding=1)

        # ---- encoder (FIX 1: two stages) ----
        self.down1 = nn.Conv3d(z_ch, z_ch, 4, stride=2, padding=1)
        self.enc1 = ResBlock3D(z_ch, tdim)
        self.down2 = nn.Conv3d(z_ch, z_ch, 4, stride=2, padding=1)
        self.enc2 = ResBlock3D(z_ch, tdim)

        # ---- bottleneck (FIX 1: two blocks) ----
        self.mid1 = ResBlock3D(z_ch, tdim)
        self.mid2 = ResBlock3D(z_ch, tdim)

        if use_temporal_suite:
            self.temporal_suite = TimeGatedConvSuite(z_ch)
        else:
            self.temporal_suite = None

        # ---- decoder (FIX 1 + FIX 3 concat skips) ----
        self.up2 = nn.ConvTranspose3d(z_ch, z_ch, 4, stride=2, padding=1)
        # concat with enc1 output (z_ch) -> 2*z_ch in
        self.dec2 = ResBlock3D(z_ch * 2, tdim, out_ch=z_ch)

        self.up1 = nn.ConvTranspose3d(z_ch, z_ch, 4, stride=2, padding=1)
        # concat with in_conv output x1 (z_ch) -> 2*z_ch in
        self.dec1 = ResBlock3D(z_ch * 2, tdim, out_ch=z_ch)

        # ---- output ----
        self.out = nn.Conv3d(z_ch, z_ch, 3, padding=1)

    def forward(self, z, t, cond=None, alpha=1.0):
        """
        z:    (B, C, D, H, W)
        cond: (B, cond_ch, D, H, W)
        t:    (B,)
        """
        # ---- FIX 6: normalize input ----
        if self.normalize_input:
            z = z / (z.std(dim=(2, 3, 4), keepdim=True) + 1e-6)

        t_emb = self.time_embed(t)
        # ---- FIX 2: concat conditioning at input ----
        if cond is not None:
            # 🔹 input-level conditioning
            cond_in = F.interpolate(
                cond,
                size=z.shape[2:],
                mode="trilinear",
                align_corners=False
            )
        
            # 🔹 bottleneck conditioning
            cond_mid = F.interpolate(
                cond,
                size=(z.shape[2] // 4, z.shape[3] // 4, z.shape[4] // 4),
                mode="trilinear",
                align_corners=False
            )
        
            # concat at input
            x = torch.cat([z, cond_in], dim=1)
        else:
            zeros = z.new_zeros(z.shape[0], self.cond_ch, *z.shape[2:])
            x = torch.cat([z, zeros], dim=1)
            cond_mid = None
        
        # ---- input conv ----
        x1 = self.in_conv(x)            # skip 1  (z_ch, full res)
        
        # ---- encoder ----
        e1 = self.enc1(self.down1(x1), t_emb)   # skip 2  (/2)
        e2 = self.enc2(self.down2(e1), t_emb)   #        (/4)
        
        # ---- bottleneck ----
        h = self.mid1(e2, t_emb)
        h = self.mid2(h, t_emb)
        
        # 🔥 FIX: inject cond_mid HERE
        if cond_mid is not None:
            h = h + 0.5 * cond_mid
        
        # optional temporal
        if self.temporal_suite is not None:
            h = h + self.temporal_suite(h, t)
        
        # ---- decoder ----
        h = self.up2(h)                          # -> /2
        h = torch.cat([h, e1], dim=1)
        h = self.dec2(h, t_emb)
        
        h = self.up1(h)                          # -> full res
        h = torch.cat([h, x1], dim=1)
        h = self.dec1(h, t_emb)
        
        return self.out(h)
