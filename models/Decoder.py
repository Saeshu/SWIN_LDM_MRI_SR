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

        # z: [B, 32, D/4, H/4, W/4] → [B, 16, D/2, H/2, W/2]
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(base_ch * 2, base_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(base_ch),
            nn.SiLU()
        )

        # [B, 16, D/2, H/2, W/2] → [B, 16, D, H, W]
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(base_ch),
            nn.SiLU()
        )

        self.out = nn.Conv3d(base_ch, out_ch, kernel_size=1)

    def forward(self, z, skip1=None):
        x = self.up1(z)
        x = self.up2(x)
        return self.out(x)
