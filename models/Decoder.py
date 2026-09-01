import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Spatial upsampling
# ============================================================

class SpatialUpsample3D(nn.Module):

    def __init__(self, scale_factor=2):

        super().__init__()

        self.scale_factor = scale_factor

    def forward(self, x):

        B, C, D, H, W = x.shape

        # Treat depth as batch
        x = x.permute(
            0, 2, 1, 3, 4
        ).contiguous()

        x = x.view(
            B * D,
            C,
            H,
            W
        )

        x = F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode="nearest"
        )

        _, _, H2, W2 = x.shape

        x = x.view(
            B,
            D,
            C,
            H2,
            W2
        )

        x = x.permute(
            0,
            2,
            1,
            3,
            4
        ).contiguous()

        return x


# ============================================================
# Decoder expert suite
# ============================================================

class DecoderConvSuite(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch
    ):

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

        # ----------------------------------------------------
        # Low frequency
        # ----------------------------------------------------

        low = F.avg_pool3d(
            x,
            kernel_size=(1, 3, 3),
            stride=1,
            padding=(0, 1, 1)
        )

        # ----------------------------------------------------
        # High frequency
        # ----------------------------------------------------

        high = x - low

        # ----------------------------------------------------
        # Experts
        # ----------------------------------------------------

        return [
            self.conv_low(low),
            self.conv_high(high),
            self.conv_3x3x1(x),
            self.conv_identity(x),
            self.conv_depth(x),
        ]


# ============================================================
# Decoder block
# ============================================================

class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch,
        upsample=True,
        use_routing=True
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

        # ----------------------------------------------------
        # Router
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

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

        # ====================================================
        # Upsampling
        # ====================================================

        if self.upsample_enabled:

            x = self.upsample(x)

        # ====================================================
        # Expert computation
        # ====================================================

        feats = self.conv_suite(x)

        # ====================================================
        # Routing + mixing
        # ====================================================

        if self.use_routing:

            logits = self.router(x)

            weights = F.softmax(
                logits,
                dim=1
            )

            # ------------------------------------------------
            # Avoid zeros_like allocation
            # ------------------------------------------------

            y = (
                weights[:, 0:1]
                * feats[0]
            )

            for i in range(
                1,
                self.num_kernels
            ):

                y = (
                    y
                    + weights[:, i:i+1]
                    * feats[i]
                )

        else:

            # ------------------------------------------------
            # Uniform mixing without constructing:
            #
            # [B, 5, D, H, W]
            #
            # ------------------------------------------------

            weight = (
                1.0 / self.num_kernels
            )

            y = feats[0] * weight

            for i in range(
                1,
                self.num_kernels
            ):

                y = (
                    y
                    + feats[i] * weight
                )

            weights = None

        # ====================================================
        # Normalization
        # ====================================================

        y = self.act(
            self.norm(y)
        )

        # ====================================================
        # Return
        # ====================================================

        if return_weights:

            return y, weights

        return y


# ============================================================
# Output refinement head
# ============================================================

class OutputRefinementHead(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch=1
    ):

        super().__init__()

        self.spatial = nn.Conv3d(
            in_ch,
            out_ch,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1)
        )

        self.depth = nn.Conv3d(
            in_ch,
            out_ch,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0)
        )

    def forward(self, x):

        return (
            self.spatial(x)
            + 0.3 * self.depth(x)
        )
