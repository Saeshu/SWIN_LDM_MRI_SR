#Need to add skip connections
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torch.nn.functional as F

def match_shape(x, ref):
    """
    Match x spatial shape to ref by symmetric padding or cropping.
    """
    _, _, D, H, W = ref.shape
    d, h, w = x.shape[2:]

    pd = D - d
    ph = H - h
    pw = W - w

    # Pad if too small
    if pd > 0 or ph > 0 or pw > 0:
        x = F.pad(
            x,
            (
                pw // 2, pw - pw // 2,
                ph // 2, ph - ph // 2,
                pd // 2, pd - pd // 2,
            )
        )

    # Crop if too large
    x = x[:, :, :D, :H, :W]
    return x
class Decoder3D(nn.Module):
    def __init__(self, out_ch=1, base_ch=16):
        super().__init__()

        # z: [B, 32, 30, 36, 30] → [B, 16, 60, 72, 60]
        self.up1 = nn.ConvTranspose3d(
            in_channels=base_ch * 2,
            out_channels=base_ch,
            kernel_size=2,
            stride=2
        )

        self.fuse1 = nn.Sequential(
            nn.Conv3d(base_ch + base_ch, base_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(base_ch),
            nn.SiLU()
        )

        # [B, 16, 60, 72, 60] → [B, 16, 120, 144, 120]
        self.up2 = nn.ConvTranspose3d(
            in_channels=base_ch,
            out_channels=base_ch,
            kernel_size=2,
            stride=2
        )

        # final projection to input channels
        self.out = nn.Conv3d(base_ch, out_ch, kernel_size=1)

    def forward(self, z, skip1):
        x = self.up1(z)                 # 30 → 60
        skip1 = match_shape(skip1, x)

        x = torch.cat([x, skip1], dim=1)
        x = self.fuse1(x)

        x = self.up2(x)                 # 60 → ~120
        x = self.out(x)

        return x
   

