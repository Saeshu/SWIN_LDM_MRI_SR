import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange



class AnisotropicConvSuite(nn.Module):
    def __init__(self, in_ch, out_ch, depth_kernels=(3, 5, 7)):
        super().__init__()

        self.conv_3x3x1 = nn.Conv3d(
            in_ch, out_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1)
        )

        self.depth_convs = nn.ModuleList([
            nn.Conv3d(
                in_ch, out_ch,
                kernel_size=(k, 1, 1),
                padding=(k // 2, 0, 0)
            )
            for k in depth_kernels
        ])

        self.conv_1x1x1 = nn.Conv3d(in_ch, out_ch, kernel_size=1)

        self.num_paths = 1 + len(depth_kernels) + 1

    def forward(self, x):
        feats = []

        feats.append(self.conv_3x3x1(x))

        for conv in self.depth_convs:
            feats.append(conv(x))

        feats.append(self.conv_1x1x1(x))

        return feats


class WindowPool3D(nn.Module):
    def __init__(self, window_size=(1, 7, 7)):
        super().__init__()
        self.window_size = window_size

    def forward(self, x):
        B, C, D, H, W = x.shape
        wd, wh, ww = self.window_size

        x = x.unfold(2, wd, wd) \
             .unfold(3, wh, wh) \
             .unfold(4, ww, ww)

        x = x.contiguous().view(B, C, -1, wd * wh * ww)

        # 🔥 IMPORTANT: less smoothing than pure mean
        tokens = x.mean(dim=-1) + 0.5 * x.std(dim=-1)

        tokens = tokens.permute(0, 2, 1)  # [B, N, C]

        return tokens


class KernelMixingAttention(nn.Module):
    def __init__(self, embed_dim, num_kernels, num_heads=4):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.proj = nn.Linear(embed_dim, num_kernels)

    def forward(self, tokens):
        attn_out, _ = self.attn(tokens, tokens, tokens)
        logits = self.proj(attn_out)
        weights = F.softmax(logits, dim=-1)
        return weights  # [B, N, K]


class AnisotropicSwinBlock(nn.Module):
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
        self.window_size = window_size

        if use_attention:
            self.window_pool = WindowPool3D(window_size)
            self.attn = KernelMixingAttention(
                embed_dim=in_ch,
                num_kernels=self.num_kernels
            )
        else:
            self.alpha = nn.Parameter(torch.ones(self.num_kernels))

        self.norm = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()

    def forward(self, x, return_weights=False):
        B, C, D, H, W = x.shape

        feats = self.conv_suite(x)  # list of [B, out_ch, D, H, W]

        if self.use_attention:
            tokens = self.window_pool(x)  # [B, N, C]

            weights = self.attn(tokens)  # [B, N, K]

            wd, wh, ww = self.window_size

            Nd = D // wd
            Nh = H // wh
            Nw = W // ww

            # reshape back to grid
            weights = weights.view(B, Nd, Nh, Nw, self.num_kernels)

            # move kernel dim forward
            weights = weights.permute(0, 4, 1, 2, 3)  # [B, K, Nd, Nh, Nw]

            # 🔥 CRITICAL: upsample to full resolution
            weights = F.interpolate(
                weights,
                size=(D, H, W),
                mode="trilinear",
                align_corners=False
            )  # [B, K, D, H, W]

        else:
            weights = F.softmax(self.alpha, dim=0)
            weights = weights.view(1, self.num_kernels, 1, 1, 1).expand(B, -1, D, H, W)

        # 🔥 spatially varying fusion
        y = 0
        for i, f in enumerate(feats):
            y = y + weights[:, i:i+1] * f

        y = self.act(self.norm(y))

        if return_weights:
            return y, weights

        return y


class SpatialDownsample3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool3d(
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2)
        )

    def forward(self, x):
        return self.pool(x)
