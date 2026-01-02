import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import timestep_embedding


class ConditionalEpsUNet3D(nn.Module):
    def __init__(self, z_ch, cond_ch, tdim=128):
        super().__init__()

        self.t_mlp = nn.Sequential(
            nn.Linear(tdim, z_ch),
            nn.SiLU(),
            nn.Linear(z_ch, z_ch)
        )

        self.in_conv = nn.Conv3d(z_ch + cond_ch, z_ch, 3, padding=1)
        self.mid = nn.Conv3d(z_ch, z_ch, 3, padding=1)
        self.out = nn.Conv3d(z_ch, z_ch, 3, padding=1)

    def forward(self, z, t, cond):
        """
        z:    noisy latent
        cond: conditioning latent
        t:    timestep tensor [B]
        """

        t_emb = timestep_embedding(t, self.t_mlp[0].in_features)
        t_emb = self.t_mlp(t_emb)
        t_emb = t_emb[:, :, None, None, None]

        x = torch.cat([z, cond], dim=1)
        x = self.in_conv(x)

        x = x + t_emb
        x = F.relu(self.mid(x))
        return self.out(x)
