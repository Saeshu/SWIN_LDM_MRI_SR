import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint



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
    def __init__(self, smooth=True):
        super().__init__()
        self.smooth = smooth

    def forward(self, feats, w, return_weights=False):
        assert all(f.shape[1] == feats[0].shape[1] for f in feats), \
        "All kernel outputs must have same channel dim"
        w = F.interpolate(
            w,
            size=feats[0].shape[2:],
            mode="trilinear",
            align_corners=False
        )

        w = w.float().contiguous()

        # 🔥 spatial smoothing
        # if self.smooth:
        #     w = F.avg_pool3d(w, (1,3,3), 1, (0,1,1))
        # w = F.avg_pool3d(w, kernel_size=(1,3,3), stride=1, padding=(0,1,1))
        w = F.avg_pool3d(w, (1,3,3), 1, (0,1,1)) * 0.5 + w * 0.5
        
        # 🔥 competition
        w = w - 0.5 * w.mean(dim=1, keepdim=True)

        # 🔥 safer normalization
        std = torch.clamp(w.std(dim=1, keepdim=True), min=1e-3)
        w = w / std
        w = w + 0.01 * torch.randn_like(w)
        
        w = torch.clamp(w, -3.0, 3.0)
        
        # 🔥 softmax with temperature
        temp = 2.0
        weights = F.softmax(w / temp, dim=1)

        # 🔥 mixing
        # print("weights shape:", weights.shape)
        # print("num feats:", len(feats))
        # print("feat shape:", feats[0].shape)
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
            scale_factor=(1,2,2),
            mode="trilinear",
            align_corners=False
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

        self.conv_low = nn.Conv3d(in_ch, out_ch, 1)
        self.conv_high = nn.Conv3d(in_ch, out_ch, 1)

        self.conv_3x3x1 = nn.Conv3d(in_ch, out_ch, (1,3,3), padding=(0,1,1))
        self.conv_identity = nn.Conv3d(in_ch, out_ch, 1)
        self.conv_depth = nn.Conv3d(in_ch, out_ch, (3,1,1), padding=(1,0,0))

        self.num_paths = 5

    def forward(self, x):
        low = F.avg_pool3d(x, (1,3,3), stride=1, padding=(0,1,1))
        high = x - low

        feats = [
            self.conv_low(low),        # ✅ now out_ch
            self.conv_high(high),      # ✅ now out_ch
            self.conv_3x3x1(x),
            self.conv_identity(x),
            self.conv_depth(x),
        ]

        return feats
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

        # ------------------------------------------------
        # Frequency fusion
        # ------------------------------------------------

        self.freq_fuse = nn.Conv3d(
            in_ch * 2,
            in_ch,
            kernel_size=1
        )

        # ------------------------------------------------
        # Expert bank
        # ------------------------------------------------

        self.conv_suite = DecoderConvSuite(
            in_ch,
            out_ch
        )

        self.num_kernels = (
            self.conv_suite.num_paths
        )

        # ------------------------------------------------
        # Router
        # ------------------------------------------------

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

        # ------------------------------------------------
        # Output
        # ------------------------------------------------

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

        # =================================================
        # 1. Frequency decomposition
        # =================================================

        low = F.avg_pool3d(
            x,
            kernel_size=(1, 3, 3),
            stride=1,
            padding=(0, 1, 1)
        )

        high = x - low

        # =================================================
        # 2. Fuse low + high
        # =================================================

        freq = torch.cat(
            [low, high],
            dim=1
        )

        freq = self.freq_fuse(freq)

        # =================================================
        # 3. Routing
        # =================================================

        if self.use_routing:

            logits = self.router(freq)

            w_dec = F.softmax(
                logits,
                dim=1
            )

        else:

            B, _, D, H, W = freq.shape

            w_dec = torch.full(
                (
                    B,
                    self.num_kernels,
                    D,
                    H,
                    W
                ),
                1.0 / self.num_kernels,
                device=freq.device,
                dtype=freq.dtype
            )

        # =================================================
        # 4. Expert processing
        # =================================================

        feats = self.conv_suite(freq)

        # =================================================
        # 5. Adaptive mixing
        # =================================================

        y = torch.zeros_like(
            feats[0]
        )

        for i, f in enumerate(feats):

            y = (
                y
                + w_dec[:, i:i+1] * f
            )

        # =================================================
        # 6. Normalization + activation
        # =================================================

        y = self.act(
            self.norm(y)
        )

        # =================================================
        # 7. Upsample AFTER adaptive processing
        # =================================================

        if self.upsample_enabled:

            y = F.interpolate(
                y,
                scale_factor=(1, 2, 2),
                mode="nearest"
            )

        # =================================================
        # 8. Return
        # =================================================

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
