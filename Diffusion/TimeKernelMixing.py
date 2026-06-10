import torch
import torch.nn as nn
import torch.nn.fucntional as F

class TimeKernelMixing(nn.Module):
    def __init__(self, channels, num_kernels, tdim=128):
        super().__init__()

        self.num_kernels = num_kernels

        # time embedding projection
        self.time_mlp = nn.Sequential(
            nn.Linear(tdim, channels),
            nn.SiLU(),
            nn.Linear(channels, channels)
        )

        # feature projection
        self.feat_proj = nn.Conv3d(channels, channels, kernel_size=1)

        # kernel logits
        self.to_logits = nn.Conv3d(channels, num_kernels, kernel_size=1)

    def forward(self, x, t_emb):
        """
        x: (B, C, D, H, W)
        t_emb: (B, tdim)
        """

        B, C, D, H, W = x.shape

        # inject time
        t_feat = self.time_mlp(t_emb)[:, :, None, None, None]
        h = x + t_feat

        # project
        h = self.feat_proj(h)

        # logits
        logits = self.to_logits(h)  # (B, K, D, H, W)

        # softmax across kernels
        weights = F.softmax(logits, dim=1)

        return weights
