import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange

# --------------------------------------------------
# Spatial upsampling (H, W only)
# --------------------------------------------------
class SpatialUpsample3D(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        B, C, D, H, W = x.shape

        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * D, C, H, W)

        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")

        _, _, H2, W2 = x.shape

        x = x.view(B, D, C, H2, W2)
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        return x


# --------------------------------------------------
# Decoder convolution suite
# --------------------------------------------------
class DecoderConvSuite(nn.Module):
    def __init__(self, in_ch, out_ch, use_depth=True):
        super().__init__()

        self.conv_3x3x1 = nn.Conv3d(in_ch, out_ch, (1,3,3), padding=(0,1,1))
        self.conv_1x3x1 = nn.Conv3d(in_ch, out_ch, (1,3,1), padding=(0,1,0))
        self.conv_3x1x1 = nn.Conv3d(in_ch, out_ch, (1,1,3), padding=(0,0,1))

        self.use_depth = use_depth

        if use_depth:
            self.conv_1x1x3 = nn.Conv3d(in_ch, out_ch, (3,1,1), padding=(1,0,0))

        self.conv_1x1x1 = nn.Conv3d(in_ch, out_ch, 1)

        self.num_paths = 4 + (1 if use_depth else 0)

    def forward(self, x):
        feats = [
            self.conv_3x3x1(x),
            self.conv_1x3x1(x),
            self.conv_3x1x1(x),
        ]

        if self.use_depth:
            feats.append(self.conv_1x1x3(x))

        feats.append(self.conv_1x1x1(x))

        return feats


# --------------------------------------------------
# Decoder block (NOW SPATIALLY AWARE)
# --------------------------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_depth=True, enc_kernel_dim=None, upsample=True):
        super().__init__()

        self.upsample_enabled = upsample
        self.upsample = SpatialUpsample3D(scale_factor=2)

        self.conv_suite = DecoderConvSuite(in_ch, out_ch, use_depth)
        self.num_dec_kernels = self.conv_suite.num_paths

        # fallback global weights
        self.logits = nn.Parameter(torch.zeros(self.num_dec_kernels))

        self.norm = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()

    def forward(self, x, encoder_kernel_skip=None, bias_strength=1.0):

        if self.upsample_enabled:
            x = self.upsample(x)

        def heavy(x):
            feats = self.conv_suite(x)
            B, C, D, H, W = feats[0].shape

            # --------------------------------------------------
            # 🔥 CASE 1: spatial weights from encoder
            # --------------------------------------------------
            if encoder_kernel_skip is not None and encoder_kernel_skip.dim() == 5:

                weights = encoder_kernel_skip  # [B, K, D, H, W]

                # match number of kernels if needed
                if weights.shape[1] != self.num_dec_kernels:
                    weights = F.interpolate(
                        weights,
                        size=(D, H, W),
                        mode="trilinear",
                        align_corners=False
                    )

                # normalize across kernels
                weights = F.softmax(weights, dim=1)

            # --------------------------------------------------
            # 🔥 CASE 2: fallback global weights
            # --------------------------------------------------
            else:
                logits = self.logits
                weights = F.softmax(logits, dim=0)
                weights = weights.view(1, self.num_dec_kernels, 1, 1, 1)
                weights = weights.expand(B, -1, D, H, W)

            # --------------------------------------------------
            # 🔥 spatial fusion
            # --------------------------------------------------
            y = 0
            for i, f in enumerate(feats):
                y = y + weights[:, i:i+1] * f

            return y

        y = checkpoint(heavy, x, use_reentrant=False)
        return self.act(self.norm(y))


# --------------------------------------------------
# Output refinement head
# --------------------------------------------------
class OutputRefinementHead(nn.Module):
    def __init__(self, in_ch, out_ch=1):
        super().__init__()

        self.spatial = nn.Conv3d(
            in_ch, out_ch, kernel_size=(1,3,3), padding=(0,1,1)
        )

        self.depth = nn.Conv3d(
            in_ch, out_ch, kernel_size=(3,1,1), padding=(1,0,0)
        )

    def forward(self, x):
        return self.spatial(x) + 0.3 * self.depth(x)
