import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ShapedEncoder3D import *

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


class ShapedUpBlock3D(nn.Module):
    """
    Upsampling block using SWIN-guided shaped convolutions.
    Mirrors ShapedDownBlock3D but upsamples instead of downsamples.
    """
    def __init__(self, in_ch, out_ch, window_size=4):
        super().__init__()
        # Upsample first (transposed conv)
        self.upsample = nn.ConvTranspose3d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=2,
            stride=2
        )
        # Then apply shaped convolution
        self.shaped = SwinGuidedConvBlock3D(out_ch, out_ch, window_size=window_size)
        
    def forward(self, x):
        x = self.upsample(x)
        x = self.shaped(x)
        return x


class SimpleShapedDecoder3D(nn.Module):
    """
    Simple decoder for testing and validation.
    NO skip connections - pure bottleneck test.
    Uses shaped convolutions to mirror the encoder.
    """
    def __init__(self, out_ch=1, base_ch=16):
        super().__init__()
        
        # Mirrors encoder structure:
        # Encoder: 1 → 16 → 32
        # Decoder: 32 → 16 → 1
        
        # z: [B, 32, D/4, H/4, W/4] → [B, 16, D/2, H/2, W/2]
        self.up1 = ShapedUpBlock3D(base_ch * 2, base_ch)
        
        # [B, 16, D/2, H/2, W/2] → [B, 16, D, H, W]
        self.up2 = ShapedUpBlock3D(base_ch, base_ch)
        
        # Final projection to output channels
        self.out = nn.Conv3d(base_ch, out_ch, kernel_size=1)
        
    def forward(self, z):
        """
        Args:
            z: Latent from encoder, shape [B, 32, D/4, H/4, W/4]
        Returns:
            Reconstructed output, shape [B, out_ch, D, H, W]
        """
        x = self.up1(z)
        x = self.up2(x)
        x = self.out(x)
        return x


class ShapedDecoder3D(nn.Module):
    """
    Decoder with skip connections for U-Net style architecture.
    Compatible with ShapedEncoder3D - accepts encoder outputs and skip connections.
    Uses SWIN-guided shaped convolutions throughout.
    """
    def __init__(self, out_ch=1, base_ch=16, window_size=4):
        super().__init__()
        
        # Architecture mirrors encoder:
        # Encoder: 1 → 16 (skip1) → 32 (z)
        # Decoder: 32 → 16+skip1 → out_ch
        
        # z: [B, 32, D/4, H/4, W/4] → [B, 16, D/2, H/2, W/2]
        self.up1 = nn.ConvTranspose3d(
            in_channels=base_ch * 2,
            out_channels=base_ch,
            kernel_size=2,
            stride=2
        )
        
        # Fuse with skip connection using shaped convolution
        # Input: base_ch (from up1) + base_ch (from skip1) = 2*base_ch
        self.fuse1 = SwinGuidedConvBlock3D(
            in_ch=base_ch + base_ch,
            out_ch=base_ch,
            window_size=window_size
        )
        
        # [B, 16, D/2, H/2, W/2] → [B, 16, D, H, W]
        self.up2 = nn.ConvTranspose3d(
            in_channels=base_ch,
            out_channels=base_ch,
            kernel_size=2,
            stride=2
        )
        
        # Final shaped convolution before output
        self.final_conv = SwinGuidedConvBlock3D(
            in_ch=base_ch,
            out_ch=base_ch,
            window_size=window_size
        )
        
        # Output projection
        self.out = nn.Conv3d(base_ch, out_ch, kernel_size=1)
        
    def forward(self, z, skip1=None):
        """
        Args:
            z: Latent from encoder, shape [B, 32, D/4, H/4, W/4]
            skip1: Skip connection from encoder.down1, shape [B, 16, D/2, H/2, W/2]
        Returns:
            Reconstructed output, shape [B, out_ch, D, H, W]
        """
        # Upsample latent
        x = self.up1(z)
        
        # Handle skip connection
        if skip1 is None:
            skip1 = torch.zeros_like(x)
        
        # Match shapes (handles resolution mismatches)
        skip1 = match_shape(skip1, x)
        
        # Concatenate and fuse with shaped convolution
        x = torch.cat([x, skip1], dim=1)
        x = self.fuse1(x)
        
        # Final upsample
        x = self.up2(x)
        x = self.final_conv(x)
        
        # Project to output channels
        x = self.out(x)
        return x


# ============================================================================
# Integration with Your Original Architecture
# ============================================================================

class KernelBasis3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.k1 = nn.Conv3d(in_ch, out_ch, 1, padding=0)
        self.k333 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.k133 = nn.Conv3d(in_ch, out_ch, (1, 3, 3), padding=(0, 1, 1))
        #self.k313 = nn.Conv3d(in_ch, out_ch, (3, 1, 3), padding=(1, 0, 1))
        self.k331 = nn.Conv3d(in_ch, out_ch, (3, 3, 1), padding=(1, 1, 0))
        
    def forward(self, x):
        return torch.stack([
            self.k1(x),
            self.k333(x),
            self.k133(x),
            #self.k313(x),
            self.k331(x),
        ], dim=1)  # [B, 5, C_out, D, H, W]


class SwinMixingField3D(nn.Module):
    """Uses proper SWIN attention to generate kernel mixing weights."""
    def __init__(self, in_ch, num_kernels=4, window_size=4, num_heads=4, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.window_size = window_size
        self.num_heads = num_heads
        self.num_kernels = num_kernels
        
        # Embed input to num_kernels channels
        self.embed = nn.Conv3d(in_ch, num_kernels, 3, padding=1)
        
        # SWIN blocks will be created dynamically based on input resolution
        self.block1 = None
        self.block2 = None
        
    def _ensure_blocks(self, resolution):
        """Create SWIN blocks if they don't exist or resolution changed."""
        if self.block1 is None or self.block1.input_resolution != resolution:
            self.block1 = SwinTransformerBlock3D(
                dim=self.num_kernels,
                input_resolution=resolution,
                num_heads=self.num_heads,
                window_size=self.window_size,
                shift_size=0
            ).to(self.embed.weight.device)
            
            self.block2 = SwinTransformerBlock3D(
                dim=self.num_kernels,
                input_resolution=resolution,
                num_heads=self.num_heads,
                window_size=self.window_size,
                shift_size=self.window_size // 2
            ).to(self.embed.weight.device)
        
    def forward(self, x):
        B, C, D, H, W = x.shape
        resolution = (D, H, W)
        
        # Ensure blocks match current resolution
        self._ensure_blocks(resolution)
        
        alpha = self.embed(x)
        alpha = self.block1(alpha)
        alpha = self.block2(alpha)
        return torch.softmax(alpha / self.temperature, dim=1)


class AttentionShapedConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.basis = KernelBasis3D(in_ch, out_ch)
        
    def forward(self, x, alpha):
        """
        Args:
            x: (B, C_in, D, H, W)
            alpha: (B, 5, D, H, W) - kernel mixing weights
        Returns:
            out: (B, C_out, D, H, W)
        """
        basis_out = self.basis(x)  # [B, 5, C_out, D, H, W]
        alpha = alpha.unsqueeze(2)  # [B, 5, 1, D, H, W]
        out = (basis_out * alpha).sum(dim=1)  # [B, C_out, D, H, W]
        return out


class SwinGuidedConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, window_size=4):
        super().__init__()
        self.swin = SwinMixingField3D(in_ch, window_size=window_size)
        self.conv = AttentionShapedConv3D(in_ch, out_ch)
        
        # Dynamic group norm
        num_groups = min(8, out_ch) if out_ch >= 8 else out_ch
        self.norm = nn.GroupNorm(num_groups, out_ch)
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
