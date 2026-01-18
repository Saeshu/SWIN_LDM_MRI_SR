import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

# ============================================================================
# 3D Window Operations (adapted from official 2D version)
# ============================================================================

def window_partition_3d(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size, window_size, 
               H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    windows = windows.view(-1, window_size, window_size, window_size, C)
    return windows


def window_reverse_3d(windows, window_size, D, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, window_size, C)
        window_size (int): Window size
        D (int): Depth of volume
        H (int): Height of volume
        W (int): Width of volume
    Returns:
        x: (B, D, H, W, C)
    """
    B = int(windows.shape[0] / (D * H * W / window_size / window_size / window_size))
    x = windows.view(B, D // window_size, H // window_size, W // window_size,
                     window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    x = x.view(B, D, H, W, -1)
    return x


# ============================================================================
# 3D Window Attention with Relative Position Bias
# ============================================================================

class WindowAttention3D(nn.Module):
    """
    Window based multi-head self attention (W-MSA) for 3D volumes.
    Includes relative position bias like official SWIN.
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, 
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table for 3D
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1) * 
                       (2 * window_size - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size)
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_d, coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 2] += self.window_size - 1
        relative_coords[:, :, 0] *= (2 * self.window_size - 1) * (2 * self.window_size - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size - 1)
        
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww*Wd, Wh*Ww*Wd) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size ** 3, self.window_size ** 3, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ============================================================================
# 3D SWIN Transformer Block with Shifted Windows
# ============================================================================

class SwinTransformerBlock3D(nn.Module):
    """
    Swin Transformer Block for 3D volumes.
    Supports both regular and shifted window attention.
    Automatically adjusts window size if it doesn't divide resolution evenly.
    """
    def __init__(self, dim, input_resolution, num_heads, window_size=4,
                 shift_size=0, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # (D, H, W)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # Adjust window size if it doesn't fit
        D, H, W = self.input_resolution
        if D < self.window_size or H < self.window_size or W < self.window_size:
            self.window_size = min(D, H, W)
            self.shift_size = 0
            print(f"Warning: window_size adjusted to {self.window_size} for resolution {input_resolution}")
        
        # Ensure window size divides evenly
        while D % self.window_size != 0 or H % self.window_size != 0 or W % self.window_size != 0:
            self.window_size -= 1
            if self.window_size < 2:
                raise ValueError(f"Cannot find valid window size for resolution {input_resolution}")
        
        # Adjust shift size accordingly
        if self.shift_size > 0:
            self.shift_size = min(self.shift_size, self.window_size // 2)

        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = nn.Identity() if drop_path <= 0. else nn.Identity()  # Simplified
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

        # Calculate attention mask for SW-MSA
        if self.shift_size > 0:
            D, H, W = self.input_resolution
            img_mask = torch.zeros((1, D, H, W, 1))
            
            d_slices = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size, None))
            h_slices = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size, None))
            
            cnt = 0
            for d in d_slices:
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, d, h, w, :] = cnt
                        cnt += 1

            mask_windows = window_partition_3d(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size ** 3)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        """
        x: (B, C, D, H, W)
        """
        D, H, W = self.input_resolution
        B, C, D_in, H_in, W_in = x.shape
        assert D == D_in and H == H_in and W == W_in, "input feature has wrong size"

        # Convert to (B, D, H, W, C) for window operations
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        
        shortcut = x
        x = self.norm1(x)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size, -self.shift_size),
                                  dims=(1, 2, 3))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition_3d(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size ** 3, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, 
                                         self.window_size, C)

        # Reverse cyclic shift
        if self.shift_size > 0:
            shifted_x = window_reverse_3d(attn_windows, self.window_size, D, H, W)
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size, self.shift_size),
                          dims=(1, 2, 3))
        else:
            x = window_reverse_3d(attn_windows, self.window_size, D, H, W)

        # Residual connection
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # Convert back to (B, C, D, H, W)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
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
