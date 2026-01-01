#Need to add skip connections
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
class Decoder3D(nn.Module):
    def __init__(self, in_ch=64, out_ch=1, base_ch=32):
        super().__init__()

        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(in_ch, base_ch * 2, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(base_ch * 2, base_ch, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(base_ch, out_ch, kernel_size=2, stride=2)
        )

    def forward(self, z):
        z = self.up1(z)  # 8 → 16
        z = self.up2(z)  # 16 → 32
        z = self.up3(z)  # 32 → 64
        return z

