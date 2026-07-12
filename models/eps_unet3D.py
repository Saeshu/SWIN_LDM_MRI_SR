import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import timestep_embedding
from Diffusion.convsuite import TimeGatedConvSuite
#from Diffusion.schedule import SinusoidalTimeEmbedding
from Diffusion.LinearNoise import NoiseScheduler
from models.ae import WE2FiLM
#SinusoidalTimeEmbedding = SinusoidalTimeEmbedding(128)
scheduler = NoiseScheduler()

import torch
import torch.nn as nn
import torch.nn.functional as F


class WE2Conditioning(nn.Module):
    """
    Spatial conditioning using the encoder routing maps.

    Inputs
    ------
    h     : [B, C, D, H, W]
    w_e2  : [B, K, D', H', W']

    Output
    ------
    Conditioned latent
    """

    def __init__(
        self,
        latent_channels,
        num_kernels=4,
    ):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv3d(
                num_kernels,
                latent_channels,
                kernel_size=3,
                padding=1,
            ),

            nn.GroupNorm(8, latent_channels),

            nn.SiLU(),

            nn.Conv3d(
                latent_channels,
                latent_channels,
                kernel_size=3,
                padding=1,
            ),
        )

        # Learn conditioning strength
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        h,
        w_e2,
    ):

        # Match bottleneck resolution
        w = F.interpolate(
            w_e2,
            size=h.shape[2:],
            mode="trilinear",
            align_corners=False,
        )

        correction = self.net(w)

        return h + self.scale * correction

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

        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)

        self.conv1 = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )

        self.conv2 = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )

        ####################################################
        # Zero init final conv (diffusion trick)
        ####################################################

        nn.init.zeros_(self.conv2.weight)

        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

        ####################################################

        self.time_mlp = TimeMLP(tdim, channels)

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        h = h + self.time_mlp(t_emb)
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return x + h

class ConditionalEpsUNet3D(nn.Module):
    def __init__(
        self,
        z_ch,
        cond_ch,
        alpha_bars,
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

        ####################################################
        # Diffusion schedule
        ####################################################

        self.register_buffer(
            "alpha_bars",
            alpha_bars.float().clone(),
        )

        ####################################################
        # WE2 conditioning
        ####################################################

        self.we2_cond = WE2Conditioning(
            latent_channels=z_ch,
            num_kernels=5,
        )

        ####################################################
        # Time embedding
        ####################################################

        self.time_embed = SinusoidalTimeEmbedding(tdim)

        ####################################################
        # Input
        ####################################################

        self.in_conv = nn.Conv3d(
            z_ch,
            z_ch,
            kernel_size=3,
            padding=1,
        )

        ####################################################
        # Encoder
        ####################################################

        self.down = nn.Conv3d(
            z_ch,
            z_ch,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.enc_block = ResBlock3D(
            z_ch,
            tdim,
        )

        ####################################################
        # Bottleneck
        ####################################################

        self.mid_block = ResBlock3D(
            z_ch,
            tdim,
        )

        ####################################################
        # Optional temporal module
        ####################################################

        if use_temporal_suite:
            self.temporal_suite = TimeGatedConvSuite(z_ch)
        else:
            self.temporal_suite = None

        ####################################################
        # Decoder
        ####################################################

        self.up = nn.ConvTranspose3d(
            z_ch,
            z_ch,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.dec_block = ResBlock3D(
            z_ch,
            tdim,
        )

        ####################################################
        # Output
        ####################################################

        self.out = nn.Conv3d(
            z_ch,
            z_ch,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        z,
        t,
        cond=None,
        w_e2=None,
    ):

        ####################################################
        # Time embedding
        ####################################################

        t_emb = timestep_embedding(
            t,
            self.tdim,
        )

        ####################################################
        # Conditioning strength
        ####################################################

        alpha_bar = self.alpha_bars[t].view(
            -1,
            1,
            1,
            1,
            1,
        )

        sqrt_signal = alpha_bar.sqrt()
        sqrt_noise = (1.0 - alpha_bar).sqrt()

        gamma = (
            sqrt_noise
            /
            (
                sqrt_signal
                + sqrt_noise
                + 1e-8
            )
        )

        # Optional safety clamp
        gamma = gamma.clamp(
            min=0.05,
            max=0.95,
        )

        ####################################################
        # Input conditioning
        ####################################################

        x = z

        if cond is not None:

            cond_resized = F.interpolate(
                cond,
                size=z.shape[2:],
                mode="trilinear",
                align_corners=False,
            )

            x = x + gamma * cond_resized

        ####################################################
        # Input conv
        ####################################################

        x1 = self.in_conv(x)

        ####################################################
        # Encoder
        ####################################################

        x2 = self.down(x1)

        x2 = self.enc_block(
            x2,
            t_emb,
        )

        ####################################################
        # Bottleneck
        ####################################################

        h = self.mid_block(
            x2,
            t_emb,
        )

        ####################################################
        # Mid conditioning
        ####################################################

        if cond is not None:

            cond_mid = F.interpolate(
                cond,
                size=h.shape[2:],
                mode="trilinear",
                align_corners=False,
            )

            h = h + gamma * cond_mid

        ####################################################
        # WE2 conditioning
        ####################################################

        if w_e2 is not None:
            h = self.we2_cond(
                h,
                w_e2,
            )

        ####################################################
        # Temporal module
        ####################################################

        if self.temporal_suite is not None:
            h = h + self.temporal_suite(
                h,
                t,
            )

        ####################################################
        # Decoder
        ####################################################

        h = self.up(h)

        ####################################################
        # Skip connection
        ####################################################

        h = h + x1

        ####################################################
        # Final block
        ####################################################

        h = self.dec_block(
            h,
            t_emb,
        )

        ####################################################
        # Output
        ####################################################

        return self.out(h)
