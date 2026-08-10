import torch
import torch.nn as nn
import torch.nn.functional as F


class WMapper(nn.Module):

    def __init__(
        self,
        channels,
        hidden_channels=None,
    ):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = channels

        self.net = nn.Sequential(
            nn.Conv3d(
                channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.SiLU(),

            nn.Conv3d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.SiLU(),

            nn.Conv3d(
                hidden_channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(self, w_lr):

        return w_lr + self.net(w_lr)
