import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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




class TimeGatedConvSuite(nn.Module):
    def __init__(self, channels, time_dim=128):
        super().__init__()

        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, 3)  # gates for A, B, C
        )

        self.spatial = SpatialSuite(channels)
        self.mid = MidSliceSuite(channels)
        self.long = LongSliceSuite(channels)

    def forward(self, x, t):
        """
        x: (B, C, H, W, D)
        t: (B,) diffusion timestep
        """

        # --- compute gates ---
        te = self.time_embed(t)
        gates = self.time_mlp(te)           # (B, 3)
        gates = torch.softmax(gates, dim=-1)

        gA, gB, gC = gates[:, 0], gates[:, 1], gates[:, 2]

        # reshape for broadcasting
        gA = gA[:, None, None, None, None]
        gB = gB[:, None, None, None, None]
        gC = gC[:, None, None, None, None]

        # --- specialist outputs ---
        out = (
            gA * self.spatial(x) +
            gB * self.mid(x) +
            gC * self.long(x)
        )

        return out


    def forward(self, x):
        return self.net(x)


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

