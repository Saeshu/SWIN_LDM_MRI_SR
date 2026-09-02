import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Spatial upsampling
# H/W only — depth is unchanged
# ============================================================
def mem(tag):
    torch.cuda.synchronize()
    print(
        tag,
        f"alloc={torch.cuda.memory_allocated()/1024**3:.3f} GB",
        f"peak={torch.cuda.max_memory_allocated()/1024**3:.3f} GB"
    )

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

    def forward_sequential(self, x):

        def mem(tag):
            torch.cuda.synchronize()
            print(
                f"{tag:<25}"
                f"alloc={torch.cuda.memory_allocated()/1024**3:.3f} GB  "
                f"peak={torch.cuda.max_memory_allocated()/1024**3:.3f} GB"
            )
    
        low = F.avg_pool3d(
            x,
            kernel_size=(1, 3, 3),
            stride=1,
            padding=(0, 1, 1)
        )
    
        high = x - low
    
    
        y = self.conv_low(low)
    
        feat = self.conv_high(high)
    
        y = y + feat
        del feat
    
        feat = self.conv_spatial(x)
    
        y = y + feat
        del feat
    
        feat = self.conv_point(x)
    
        y = y + feat
        del feat
    
        feat = self.conv_depth(x)
    
        y = y + feat
        del feat
    
        return y / self.num_paths


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
        channel_reduction=True,
    ):
        super().__init__()

        self.upsample_enabled = upsample
        self.use_routing = use_routing
        self.channel_reduction_enabled = channel_reduction

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
        # Five decoder experts
        # ====================================================

        self.conv_suite = DecoderConvSuite(
            reduced_ch,
            reduced_ch
        )

        self.num_kernels = self.conv_suite.num_paths

        # ====================================================
        # Router
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
        # Upsampling
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
        # 1. Channel reduction
        # ====================================================
    
        if self.channel_reduction_enabled:
            x = self.channel_down(x)
        # ====================================================
        # 2–3. Expert computation + routing
        # ====================================================
    
        if self.use_routing:

            feats = self.conv_suite(x)
        
            logits = self.router(x)
            weights = F.softmax(logits, dim=1)
        
            y = weights[:, 0:1] * feats[0]
        
            for i in range(1, self.num_kernels):
                y = y + weights[:, i:i+1] * feats[i]
        else:
        
            y = self.conv_suite.forward_sequential(x)
            weights = None
        # ====================================================
        # 4. Channel expansion
        # ====================================================
    
        if self.channel_reduction_enabled:
            y = self.channel_up(y)
        # ====================================================
        # 5. Normalization
        # ====================================================
    
        y = self.norm(y)
      
        y = self.act(y)
   
        # ====================================================
        # 6. Upsampling
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
