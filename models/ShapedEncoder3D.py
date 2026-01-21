import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class AnisotropicConvSuite(nn.Module):
    """
    Runs multiple anisotropic convolutions in parallel.
    All kernels see the SAME input.
    """

    def __init__(self, in_ch, out_ch, depth_kernels=(3, 5, 7)):
        super().__init__()

        # In-plane spatial conv (no depth mixing)
        self.conv_3x3x1 = nn.Conv3d(
            in_ch, out_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1)
        )

        # Depth-only convolutions (no spatial mixing)
        self.depth_convs = nn.ModuleList([
            nn.Conv3d(
                in_ch, out_ch,
                kernel_size=(k, 1, 1),
                padding=(k // 2, 0, 0)
            )
            for k in depth_kernels
        ])

        # 1x1x1 channel mixer (acts like residual control)
        self.conv_1x1x1 = nn.Conv3d(in_ch, out_ch, kernel_size=1)

        self.num_paths = 1 + len(depth_kernels) + 1  # spatial + depth + mixer

    def forward(self, x):
        """
        Returns a list of feature maps, one per kernel path.
        """
        feats = []

        feats.append(self.conv_3x3x1(x))      # spatial
        for conv in self.depth_convs:          # depth paths
            feats.append(conv(x))

        feats.append(self.conv_1x1x1(x))       # channel mixer

        return feats  # list of tensors, all same shape

class WindowPool3D(nn.Module):
    """
    Compresses local 3D windows into single tokens.
    This is the 'Swin-style compression' step.
    """

    def __init__(self, window_size=(1, 7, 7)):
        super().__init__()
        self.window_size = window_size

    def forward(self, x):
        """
        x: [B, C, D, H, W]
        returns: tokens [B, N_windows, C]
        """

        B, C, D, H, W = x.shape
        wd, wh, ww = self.window_size

        # unfold creates sliding windows
        x = x.unfold(2, wd, wd) \
             .unfold(3, wh, wh) \
             .unfold(4, ww, ww)
        # shape: [B, C, Nd, Nh, Nw, wd, wh, ww]

        x = x.contiguous().view(B, C, -1, wd * wh * ww)

        # pool inside each window → single token
        tokens = x.mean(dim=-1)   # [B, C, N_windows]
        tokens = tokens.permute(0, 2, 1)  # [B, N_windows, C]

        return tokens

class KernelMixingAttention(nn.Module):
    """
    Uses self-attention over window tokens to predict
    kernel mixing weights.
    """

    def __init__(self, embed_dim, num_kernels, num_heads=4):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.proj = nn.Linear(embed_dim, num_kernels)

    def forward(self, tokens):
        """
        tokens: [B, N_windows, C]
        returns: weights [B, N_windows, K]
        """

        # Self-attention inside windows
        attn_out, _ = self.attn(tokens, tokens, tokens)

        # Map context → kernel logits
        logits = self.proj(attn_out)

        # Softmax over kernels
        weights = F.softmax(logits, dim=-1)

        return weights


class AnisotropicSwinBlock(nn.Module):
    """
    Full block:
    - parallel anisotropic convs
    - window compression
    - self-attention for kernel weights
    - weighted fusion
    """

    def __init__(
        self,
        in_ch,
        out_ch,
        depth_kernels=(3, 5, 7),
        window_size=(1, 7, 7),
        use_attention=True
    ):
        super().__init__()

        self.conv_suite = AnisotropicConvSuite(
            in_ch, out_ch, depth_kernels
        )

        self.use_attention = use_attention
        self.num_kernels = self.conv_suite.num_paths

        if use_attention:
            self.window_pool = WindowPool3D(window_size)
            self.attn = KernelMixingAttention(
                embed_dim=in_ch,
                num_kernels=self.num_kernels
            )
        else:
            # fallback: learned global weights
            self.alpha = nn.Parameter(torch.ones(self.num_kernels))

        self.norm = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        feats = self.conv_suite(x)  # list of [B, C, D, H, W]

        if self.use_attention:
            tokens = self.window_pool(x)
            weights = self.attn(tokens)  # [B, N_windows, K]

            # collapse window weights to global (cheap + stable)
            weights = weights.mean(dim=1)  # [B, K]
        else:
            weights = F.softmax(self.alpha, dim=0).unsqueeze(0)

        # weighted fusion
        y = 0
        for i, f in enumerate(feats):
            y = y + weights[:, i].view(-1, 1, 1, 1, 1) * f

        y = self.act(self.norm(y))
        return y

# Early layers (no depth attention, cheap)
block_lvl0 = AnisotropicSwinBlock(
    in_ch=32, out_ch=32,
    depth_kernels=(),     # no depth mixing
    use_attention=False
)

# Mid layers
block_lvl2 = AnisotropicSwinBlock(
    in_ch=64, out_ch=128,
    depth_kernels=(3, 5),
    window_size=(1, 7, 7),
    use_attention=True
)

# Bottleneck
block_bottleneck = AnisotropicSwinBlock(
    in_ch=256, out_ch=256,
    depth_kernels=(3, 5, 7),
    window_size=(3, 7, 7),
    use_attention=True
)
