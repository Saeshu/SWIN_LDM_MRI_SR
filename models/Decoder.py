import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Spatial upsampling
# H/W only — depth is unchanged
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

        x = x.reshape(
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

        H2, W2 = x.shape[-2:]

        x = x.reshape(
            B,
            D,
            C,
            H2,
            W2
        )

        x = x.permute(
            0, 2, 1, 3, 4
        ).contiguous()

        return x


# ============================================================
# Decoder Expert Suite
#
# Experts operate on the tensor supplied to the suite.
# ============================================================

class DecoderConvSuite(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch
    ):
        super().__init__()

        # ----------------------------------------------------
        # Low-frequency expert
        # ----------------------------------------------------

        self.conv_low = nn.Conv3d(
            in_ch,
            out_ch,
            kernel_size=1
        )

        # ----------------------------------------------------
        # High-frequency expert
        # ----------------------------------------------------

        self.conv_high = nn.Conv3d(
            in_ch,
            out_ch,
            kernel_size=1
        )

        # ----------------------------------------------------
        # In-plane spatial expert
        # ----------------------------------------------------

        self.conv_spatial = nn.Conv3d(
            in_ch,
            out_ch,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1)
        )

        # ----------------------------------------------------
        # Pointwise expert
        # ----------------------------------------------------

        self.conv_point = nn.Conv3d(
            in_ch,
            out_ch,
            kernel_size=1
        )

        # ----------------------------------------------------
        # Depth expert
        # ----------------------------------------------------

        self.conv_depth = nn.Conv3d(
            in_ch,
            out_ch,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0)
        )

        self.num_paths = 5

    def forward(self, x):

        # ====================================================
        # Frequency decomposition
        # ====================================================

        low = F.avg_pool3d(
            x,
            kernel_size=(1, 3, 3),
            stride=1,
            padding=(0, 1, 1)
        )

        high = x - low

        # ====================================================
        # Expert paths
        # ====================================================

        return (
            self.conv_low(low),
            self.conv_high(high),
            self.conv_spatial(x),
            self.conv_point(x),
            self.conv_depth(x),
        )


# ============================================================
# Optimized Decoder Block
# ============================================================

class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch,
        upsample=True,
        use_routing=True,
        spatial_reduction=True,
        channel_reduction=True,
    ):
        super().__init__()

        self.upsample_enabled = upsample
        self.use_routing = use_routing
        self.spatial_reduction_enabled = spatial_reduction
        self.channel_reduction_enabled = channel_reduction

        # ====================================================
        # Spatial reduction
        #
        # Experts operate at reduced H/W.
        # ====================================================

        if self.spatial_reduction_enabled:

            self.reduce_spatial = nn.AvgPool3d(
                kernel_size=(1, 2, 2),
                stride=(1, 2, 2)
            )

        # ====================================================
        # Channel bottleneck
        # ====================================================

        if self.channel_reduction_enabled:

            reduced_ch = max(
                out_ch,
                in_ch // 2
            )

            self.reduced_ch = reduced_ch

            self.channel_down = nn.Conv3d(
                in_ch,
                reduced_ch,
                kernel_size=1
            )

        else:

            reduced_ch = in_ch
            self.reduced_ch = in_ch

        # ====================================================
        # Five experts
        # ====================================================

        self.conv_suite = DecoderConvSuite(
            reduced_ch,
            reduced_ch
        )

        self.num_kernels = (
            self.conv_suite.num_paths
        )

        # ====================================================
        # Router
        #
        # Router operates at reduced resolution.
        # ====================================================

        if self.use_routing:

            router_hidden = max(
                16,
                reduced_ch // 2
            )

            self.router = nn.Sequential(

                nn.Conv3d(
                    reduced_ch,
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

        # ====================================================
        # Channel expansion
        # ====================================================

        if self.channel_reduction_enabled:

            self.channel_up = nn.Conv3d(
                reduced_ch,
                out_ch,
                kernel_size=1
            )

        # ====================================================
        # Output
        # ====================================================

        self.norm = nn.GroupNorm(
            8,
            out_ch
        )

        self.act = nn.SiLU()

        # ====================================================
        # Upsampler
        # ====================================================

        if self.upsample_enabled:

            self.upsample = SpatialUpsample3D(
                scale_factor=2
            )

    def forward(
        self,
        x,
        return_weights=False
    ):

        # ====================================================
        # 1. Spatial reduction
        # ====================================================

        if self.spatial_reduction_enabled:

            x = self.reduce_spatial(x)

        # ====================================================
        # 2. Channel reduction
        # ====================================================

        if self.channel_reduction_enabled:

            x = self.channel_down(x)

        # ====================================================
        # 3. Expert computation
        # ====================================================

        feats = self.conv_suite(x)

        # ====================================================
        # 4. Routing
        # ====================================================

        if self.use_routing:

            logits = self.router(x)

            weights = F.softmax(
                logits,
                dim=1
            )

            # ------------------------------------------------
            # Avoid zeros_like()
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
            # Uniform mixing.
            #
            # No routing tensor is allocated.
            # ------------------------------------------------

            y = feats[0]

            for i in range(
                1,
                self.num_kernels
            ):

                y = y + feats[i]

            y = y / self.num_kernels

            weights = None

        # ====================================================
        # 5. Channel expansion
        # ====================================================

        if self.channel_reduction_enabled:

            y = self.channel_up(y)

        # ====================================================
        # 6. Normalization
        # ====================================================

        y = self.norm(y)

        y = self.act(y)

        # ====================================================
        # 7. Upsampling
        # ====================================================

        if self.upsample_enabled:

            y = self.upsample(y)

        # ====================================================
        # Return
        # ====================================================

        if return_weights:

            return y, weights

        return y


# ============================================================
# Output Refinement Head
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
