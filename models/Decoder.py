import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialUpsample3D(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * D, C, H, W)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        _, _, H2, W2 = x.shape
        x = x.view(B, D, C, H2, W2).permute(0, 2, 1, 3, 4).contiguous()
        return x


class DecoderConvSuite(nn.Module):
    def __init__(self, in_ch, out_ch, use_depth=True):
        super().__init__()
        self.conv_3x3x1 = nn.Conv3d(in_ch, out_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.conv_1x3x1 = nn.Conv3d(in_ch, out_ch, kernel_size=(1, 3, 1), padding=(0, 1, 0))
        self.conv_3x1x1 = nn.Conv3d(in_ch, out_ch, kernel_size=(1, 1, 3), padding=(0, 0, 1))
        self.use_depth = use_depth
        if use_depth:
            self.conv_1x1x3 = nn.Conv3d(in_ch, out_ch, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.conv_1x1x1 = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        self.num_paths = 4 + (1 if use_depth else 0)

    def forward(self, x):
        feats = [self.conv_3x3x1(x), self.conv_1x3x1(x), self.conv_3x1x1(x)]
        if self.use_depth:
            feats.append(self.conv_1x1x3(x))
        feats.append(self.conv_1x1x1(x))
        return feats


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_depth=True, enc_kernel_dim=None,
                 upsample=True, mix_temperature=0.7):
        super().__init__()
        self.upsample_enabled = upsample
        self.upsample = SpatialUpsample3D(scale_factor=2)
        self.conv_suite = DecoderConvSuite(in_ch=in_ch, out_ch=out_ch, use_depth=use_depth)
        self.num_dec_kernels = self.conv_suite.num_paths
        self.logits = nn.Parameter(torch.zeros(self.num_dec_kernels))
        self.mix_temperature = mix_temperature  # FIX 5

        if enc_kernel_dim is not None:
            self.enc_to_dec = nn.Linear(enc_kernel_dim, self.num_dec_kernels)
        else:
            self.enc_to_dec = None

        self.norm = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()

    def forward(self, x, encoder_kernel_skip=None):
        if self.upsample_enabled:
            x = self.upsample(x)
        return self.heavy(x, encoder_kernel_skip)

    def heavy(self, x, encoder_kernel_skip):
        feats = self.conv_suite(x)
        B, C, D, H, W = feats[0].shape
        T = self.mix_temperature

        if encoder_kernel_skip is not None and encoder_kernel_skip.dim() == 5:
            weights = F.avg_pool3d(encoder_kernel_skip, kernel_size=(1, 4, 4), stride=(1, 4, 4))
            weights = F.interpolate(weights, size=(D, H, W), mode="trilinear", align_corners=False)
            K = min(weights.shape[1], self.num_dec_kernels)
            weights = weights[:, :K]
            feats = feats[:K]
            weights = F.softmax(weights / T, dim=1)   # FIX 5: temperature
        else:
            weights = F.softmax(self.logits / T, dim=0)  # FIX 5: temperature
            weights = weights.view(1, self.num_dec_kernels, 1, 1, 1).expand(B, -1, D, H, W)

        y = torch.zeros_like(feats[0])
        for i, f in enumerate(feats):
            assert weights.shape[1] > i, f"Weight index {i} out of bounds"
            y = y + weights[:, i:i+1] * f
        # apply norm + activation so the block is self-contained
        y = self.act(self.norm(y))
        return y


class OutputRefinementHead(nn.Module):
    def __init__(self, in_ch, out_ch=1):
        super().__init__()
        self.spatial = nn.Conv3d(in_ch, out_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.depth = nn.Conv3d(in_ch, out_ch, kernel_size=(3, 1, 1), padding=(1, 0, 0))

    def forward(self, x):
        return self.spatial(x) + 0.3 * self.depth(x)
