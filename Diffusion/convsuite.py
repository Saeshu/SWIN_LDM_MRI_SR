import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.utils import timestep_embedding
#from Diffusion.convsuite import TimeGatedConvSuite
from Diffusion.schedule import SinusoidalTimeEmbedding
from Diffusion.LinearNoise import NoiseScheduler

class SpatialSuite(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(c, c, kernel_size=(3,3,1), padding=(1,1,0)),
            nn.SiLU(),
            nn.Conv3d(c, c, kernel_size=(1,3,1), padding=(0,1,0)),
            nn.SiLU(),
            nn.Conv3d(c, c, kernel_size=(3,1,1), padding=(1,0,0)),
        )

    def forward(self, x):
        return self.net(x)

class MidSliceSuite(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(c, c, kernel_size=(1,1,3), padding=(0,0,1)),
            nn.SiLU(),
            nn.Conv3d(c, c, kernel_size=(1,1,5), padding=(0,0,2)),
        )

    def forward(self, x):
        return self.net(x)

class LongSliceSuite(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(c, c, kernel_size=(1,1,7), padding=(0,0,3)),
            nn.SiLU(),
            nn.Conv3d(c, c, kernel_size=(1,1,9), padding=(0,0,4)),
        )
    def forward(self, x):
        return self.net(x)





class TimeGatedConvSuite(nn.Module):
    def __init__(self, channels, time_dim=128):
        super().__init__()
        self.prior_proj = nn.Conv3d(64, 3, kernel_size=1)
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
        nn.Linear(time_dim, channels),
        nn.SiLU(),
        nn.Linear(channels, channels)
        )

        self.to_logits = nn.Conv3d(channels, 3, kernel_size=1)

        self.spatial = SpatialSuite(channels)
        self.mid = MidSliceSuite(channels)
        self.long = LongSliceSuite(channels)

    def forward(self, x, t, encoder_prior=None):
        B, C, D, H, W = x.shape

        # ---- time embedding ----
        te = self.time_embed(t)
        t_feat = self.time_mlp(te)[:, :, None, None, None]

        # inject time into features
        h = x + t_feat

        # ---- compute logits from features ----
        logits = self.to_logits(h)   # (B, 3, D, H, W)

        # ---- inject encoder prior ----
        if encoder_prior is not None:
            prior = F.interpolate(
            encoder_prior,
            size=h.shape[2:],   # match spatial dims of h
            mode="trilinear",
            align_corners=False
        )
            print("Prior received:", prior.shape)
            prior_logits = self.prior_proj(prior)  # (B, 3, D, H, W)
            logits = logits + 3.0 * prior_logits

        # ---- convert to weights ----
        weights = torch.softmax(logits, dim=1)

        # ---- split weights ----
        wA = weights[:, 0:1]
        wB = weights[:, 1:2]
        wC = weights[:, 2:3]

        # ---- specialist convs ----
        out = (
            wA * self.spatial(x) +
            wB * self.mid(x) +
            wC * self.long(x)
        )
        print(weights[0, :, D//2, H//2, W//2])
        return out

    

class BottleneckBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.conv = nn.Conv3d(channels, channels, 3, padding=1)
        self.temporal_suite = TimeGatedConvSuite(channels)

    def forward(self, x, t):
        h = self.norm(x)
        h = F.silu(h)
        h = self.conv(h)

        # temporal / slice-aware correction
        h = h + self.temporal_suite(h, t)

        return h

