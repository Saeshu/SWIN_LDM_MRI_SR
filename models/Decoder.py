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

class SpatialUpsample3D(nn.Module):
    """
    Safe upsampling:
    - Only upsamples H and W
    - Keeps depth unchanged
    """

    def __init__(self, scale_factor=2, mode="trilinear"):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(
            x,
            scale_factor=(1, self.scale_factor, self.scale_factor),
            mode=self.mode,
            align_corners=False if self.mode != "nearest" else None
        )


class DecoderConvSuite(nn.Module):
    """
    Anisotropic convolution suite for reconstruction.
    Spatial kernels dominate.
    """

    def __init__(
        self,
        in_ch,
        out_ch,
        use_depth=True
    ):
        super().__init__()

        # Spatial refinement (main workhorses)
        self.conv_3x3x1 = nn.Conv3d(
            in_ch, out_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1)
        )
        self.conv_1x3x1 = nn.Conv3d(
            in_ch, out_ch, kernel_size=(1, 3, 1), padding=(0, 1, 0)
        )
        self.conv_3x1x1 = nn.Conv3d(
            in_ch, out_ch, kernel_size=(1, 1, 3), padding=(0, 0, 1)
        )

        self.use_depth = use_depth

        if use_depth:
            # Short-range depth consistency only
            self.conv_1x1x3 = nn.Conv3d(
                in_ch, out_ch, kernel_size=(3, 1, 1), padding=(1, 0, 0)
            )

        # Channel mixer
        self.conv_1x1x1 = nn.Conv3d(in_ch, out_ch, kernel_size=1)

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

class DecoderKernelMixer(nn.Module):
    def __init__(self, num_kernels):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_kernels))

    def forward(self, feats, encoder_bias=None, strength=1.0):
        """
        feats: list of [B, C, D, H, W]
        encoder_bias: [B, K_enc] or None
        """

        logits = self.logits

        if encoder_bias is not None:
            # global bias from encoder intent
            logits = logits + strength * encoder_bias.mean(dim=0)[:logits.numel()]

        weights = F.softmax(logits, dim=0)

        y = 0
        for w, f in zip(weights, feats):
            y = y + w * f

        return y

