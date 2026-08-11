import torch
import torch.nn as nn
import torch.nn.functional as F


class WMapper(nn.Module):
    """
    Maps LR routing weights -> decoder-compatible routing weights.

    Input:
        w_lr: [B, K, D, H, W]
               already normalized by softmax

    Output:
        w_pred: [B, K, D, H, W]
                valid routing distribution
    """

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

        # Start with approximately zero correction.
        # This makes the initial mapper behave roughly like:
        # w_pred ≈ w_lr
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, w_lr):

        # --------------------------------------------------
        # Numerical safety
        # --------------------------------------------------

        w_lr = torch.clamp(
            w_lr,
            min=1e-6,
            max=1.0,
        )

        # --------------------------------------------------
        # Treat log probabilities as routing logits
        # --------------------------------------------------

        logits_lr = torch.log(w_lr)

        # --------------------------------------------------
        # Predict a correction in logit space
        # --------------------------------------------------

        delta = self.net(w_lr)

        logits_pred = logits_lr + delta

        # --------------------------------------------------
        # Convert back to routing probabilities
        # --------------------------------------------------

        w_pred = F.softmax(
            logits_pred,
            dim=1,
        )

        return w_pred
