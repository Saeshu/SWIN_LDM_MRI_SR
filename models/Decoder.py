import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange
# from torch.utils.checkpoint import checkpoint



class SpatialKernelMixer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feats, w, return_weights=False):

        w = F.interpolate(w, size=feats[0].shape[2:], mode="trilinear", align_corners=False)
        
        weights = F.softmax(w * 5.0, dim=1)
        # print("w shape:", w.shape)
        # print("num feats:", len(feats))
        # print("feat shape:", feats[0].shape)
        y = torch.zeros_like(feats[0])
        for i, f in enumerate(feats):
            y = y + weights[:, i:i+1] * f
    
        if return_weights:
            return y, weights
    
        return y
    
class SmoothedSpatialKernelMixer(SpatialKernelMixer):
    def __init__(self, smooth=False, topk=2, temp=1.5):
        super().__init__()
        self.smooth = smooth
        self.topk = topk
        self.temp = temp

    def forward(self, feats, w, return_weights=False):
        assert all(f.shape[1] == feats[0].shape[1] for f in feats), \
            "All kernel outputs must have same channel dim"

        # -----------------------------
        # Match spatial resolution
        # -----------------------------
        w = F.interpolate(
            w,
            size=feats[0].shape[2:],
            mode="trilinear",
            align_corners=False
        ).float().contiguous()

        # -----------------------------
        # (OPTIONAL) smoothing — OFF by default
        # -----------------------------
        if self.smooth:
            w = F.avg_pool3d(w, (1,3,3), 1, (0,1,1))

        # -----------------------------
        # Competition (center across kernels)
        # -----------------------------
        w = w - w.mean(dim=1, keepdim=True)

        # -----------------------------
        # Normalize across kernels
        # -----------------------------
        std = torch.clamp(w.std(dim=1, keepdim=True), min=1e-3)
        w = w / std

        # -----------------------------
        # Break symmetry (important)
        # -----------------------------
        w = w + 0.01 * torch.randn_like(w)

        # -----------------------------
        # Clamp for stability
        # -----------------------------
        w = torch.clamp(w, -3.0, 3.0)

        # -----------------------------
        # Sharper softmax (CRITICAL)
        # -----------------------------
        weights = F.softmax(w / self.temp, dim=1)

        # -----------------------------
        # 🔥 TOP-K ROUTING (REAL FIX)
        # -----------------------------
        if self.topk is not None:
            vals, idx = torch.topk(weights, k=self.topk, dim=1)

            mask = torch.zeros_like(weights)
            mask.scatter_(1, idx, 1.0)

            weights = weights * mask
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)

        # -----------------------------
        # 🔥 Mixing (residual-style — better)
        # -----------------------------
        y = torch.zeros_like(feats[0])

        for i, f in enumerate(feats):
            y = y + weights[:, i:i+1] * f

        if return_weights:
            return y, weights

        return y
# --------------------------------------------------
# Spatial upsampling (H, W only)
# --------------------------------------------------
class SpatialUpsample3D(nn.Module):
    """
    Memory-safe spatial upsampling:
    - Upsamples H/W only
    - Treats depth as batch
    """

    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        """
        x: [B, C, D, H, W]
        """
        B, C, D, H, W = x.shape

        # Treat depth as batch
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, D, C, H, W]
        x = x.view(B * D, C, H, W)                # [B·D, C, H, W]

        # 2D upsample (cheap & safe)
        x = F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode="nearest"
        )

        _, _, H2, W2 = x.shape

        # Restore 3D structure
        x = x.view(B, D, C, H2, W2)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, D, H2, W2]

        return x



# --------------------------------------------------
# Decoder convolution suite (reconstruction-focused)
# --------------------------------------------------

class DecoderConvSuite(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv_low = nn.Conv3d(
            in_ch,
            out_ch,
            kernel_size=1
        )

        self.conv_high = nn.Conv3d(
            in_ch,
            out_ch,
            kernel_size=1
        )

        self.conv_3x3x1 = nn.Conv3d(
            in_ch,
            out_ch,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1)
        )

        self.conv_identity = nn.Conv3d(
            in_ch,
            out_ch,
            kernel_size=1
        )

        self.conv_depth = nn.Conv3d(
            in_ch,
            out_ch,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0)
        )

        self.num_paths = 5

    def forward(self, x):

        low = F.avg_pool3d(
            x,
            kernel_size=(1, 3, 3),
            stride=1,
            padding=(0, 1, 1)
        )

        high = x - low

        return [
            self.conv_low(low),
            self.conv_high(high),
            self.conv_3x3x1(x),
            self.conv_identity(x),
            self.conv_depth(x),
        ]
# --------------------------------------------------
# Decoder block (upsample + conv suite + kernel mixing)
# --------------------------------------------------

class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch,
        upsample=True,
        use_routing=True,
    ):
        super().__init__()

        self.upsample_enabled = upsample
        self.use_routing = use_routing

        self.upsample = SpatialUpsample3D(
            scale_factor=2
        )

        self.conv_suite = DecoderConvSuite(
            in_ch,
            out_ch
        )

        self.num_kernels = (
            self.conv_suite.num_paths
        )

        if self.use_routing:

            router_hidden = max(
                32,
                in_ch // 2
            )

            self.router = nn.Sequential(
                nn.Conv3d(
                    in_ch,
                    router_hidden,
                    kernel_size=3,
                    padding=1
                ),

                nn.SiLU(),

                nn.Conv3d(
                    router_hidden,
                    self.num_kernels,
                    kernel_size=1
                )
            )

        self.norm = nn.GroupNorm(
            8,
            out_ch
        )

        self.act = nn.SiLU()

    def forward(
        self,
        x,
        return_weights=False
    ):

        # ---------------------------------------------
        # Upsample
        # ---------------------------------------------

        if self.upsample_enabled:
            x = self.upsample(x)

        # ---------------------------------------------
        # Experts
        # ---------------------------------------------

        feats = self.conv_suite(x)

        # ---------------------------------------------
        # Routing + mixing
        # ---------------------------------------------

        if self.use_routing:

            logits = self.router(x)

            w_dec = F.softmax(
                logits,
                dim=1
            )

            y = (
                w_dec[:, 0:1]
                * feats[0]
            )

            for i in range(
                1,
                self.num_kernels
            ):

                y = (
                    y
                    + w_dec[:, i:i+1]
                    * feats[i]
                )

        else:

            weight = 1.0 / self.num_kernels

            y = feats[0] * weight

            for i in range(
                1,
                self.num_kernels
            ):

                y = (
                    y
                    + feats[i] * weight
                )

            w_dec = None

        # ---------------------------------------------
        # Norm + activation
        # ---------------------------------------------

        y = self.act(
            self.norm(y)
        )

        # ---------------------------------------------
        # Return
        # ---------------------------------------------

        if return_weights:

            return y, w_dec

        return y
# --------------------------------------------------
# Output refinement head (image space)
# --------------------------------------------------

class OutputRefinementHead(nn.Module):
    """
    Final reconstruction head.
    Converts decoder features into image space.
    """

    def __init__(self, in_ch, out_ch=1):
        super().__init__()

        # Spatial sharpening
        self.spatial = nn.Conv3d(
            in_ch,
            out_ch,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1)
        )

        # Gentle depth consistency
        self.depth = nn.Conv3d(
            in_ch,
            out_ch,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0)
        )

    def forward(self, x):
        return self.spatial(x) + 0.3 * self.depth(x)
