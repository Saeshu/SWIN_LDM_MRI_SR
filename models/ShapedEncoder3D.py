import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class KernelBasis3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.k1 = nn.Conv3d(in_ch, out_ch, 1, padding=0)

        self.k333 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        
        self.k133 = nn.Conv3d(in_ch, out_ch, (1, 3, 3), padding=(0, 1, 1))
        self.k313 = nn.Conv3d(in_ch, out_ch, (3, 1, 3), padding=(1, 0, 1))
        self.k331 = nn.Conv3d(in_ch, out_ch, (3, 3, 1), padding=(1, 1, 0))

    def forward(self, x):
        return torch.stack([
            self.k1(x),
            self.k333(x),
            self.k133(x),
            self.k313(x),
            self.k331(x),
            
        ], dim=1)
        # shape: [B, 5, C, D, H, W]

class SwinBlock3D(nn.Module):
    """
    Minimal Swin-style block for 3D volumes.
    - No window shifting yet
    - No multi-head complexity
    - Stable on CPU
    """

    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.Linear(dim, dim)   # token mixing
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        """
        x: [B, C, D, H, W]
        """
        B, C, D, H, W = x.shape

        # flatten spatial dims → tokens
        x_tokens = rearrange(x, 'b c d h w -> b (d h w) c')

        # attention-like mixing
        x_tokens = x_tokens + self.attn(self.norm1(x_tokens))

        # MLP
        x_tokens = x_tokens + self.mlp(self.norm2(x_tokens))

        # reshape back
        x = rearrange(x_tokens, 'b (d h w) c -> b c d h w',
                      d=D, h=H, w=W)
        return x
        
class SwinMixingField3D(nn.Module):
    def __init__(self, in_ch, num_kernels=5):
        super().__init__()
        self.embed = nn.Conv3d(in_ch, num_kernels, 3, padding=1)
        self.block1 = SwinBlock3D(num_kernels)
        self.block2 = SwinBlock3D(num_kernels)

    def forward(self, x):
        alpha = self.embed(x)
        alpha = self.block1(alpha)
        alpha = self.block2(alpha)
        return torch.softmax(alpha, dim=1)

class AttentionShapedConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.basis = KernelBasis3D(in_ch, out_ch)

    def forward(self, x, alpha):
        # basis_out: [B, 4, C, D, H, W]
        basis_out = self.basis(x)

        # alpha: [B, 4, D, H, W] → broadcast
        alpha = alpha.unsqueeze(2)

        out = (basis_out * alpha).sum(dim=1)
        return out
class SwinGuidedConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.swin = SwinMixingField3D(in_ch)
        self.conv = AttentionShapedConv3D(in_ch, out_ch)
        self.norm = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        alpha = self.swin(x)
        x = self.conv(x, alpha)
        return self.act(self.norm(x))
class ShapedDownBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.shaped = SwinGuidedConvBlock3D(in_ch, out_ch)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.shaped(x)
        x = self.pool(x)
        return x
class ShapedEncoder3D(nn.Module):
    def __init__(self, in_ch=1, base_ch=16):
        super().__init__()

        self.down1 = ShapedDownBlock3D(in_ch, base_ch)
        self.down2 = ShapedDownBlock3D(base_ch, base_ch * 2)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        return x2, x1



          
