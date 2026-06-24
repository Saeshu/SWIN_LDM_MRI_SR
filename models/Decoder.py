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
    def __init__(self, smooth=False, topk=2, temp=0.4):
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

        # 🔥 ALL must output out_ch
        self.conv_low = nn.Conv3d(in_ch, out_ch, 1)
        self.conv_high = nn.Conv3d(in_ch, out_ch, 1)

        self.conv_3x3x1 = nn.Conv3d(in_ch, out_ch, (1,3,3), padding=(0,1,1))
        self.conv_identity = nn.Conv3d(in_ch, out_ch, 1)
        self.conv_depth = nn.Conv3d(in_ch, out_ch, (3,1,1), padding=(1,0,0))

    def forward(self, x):
        low = F.avg_pool3d(x, (1,3,3), stride=1, padding=(0,1,1))
        high = x - low

        feats = [
            self.conv_low(low),
            self.conv_high(high),
            self.conv_3x3x1(x),
            self.conv_identity(x),
            self.conv_depth(x),
        ]

        return feats
# --------------------------------------------------
# Decoder block (upsample + conv suite + kernel mixing)
# --------------------------------------------------

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, upsample=True):
        super().__init__()

        self.upsample_enabled = upsample
        self.upsample = SpatialUpsample3D(scale_factor=2)

        self.conv_suite = DecoderConvSuite(in_ch, out_ch)

        # 🔥 plug-in mixer
        self.mixer = SmoothedSpatialKernelMixer()

        self.norm = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()

    def forward(self, x, w_E2=None, return_weights=False):

        if self.upsample_enabled:
            x = self.upsample(x)
    
        feats = self.conv_suite(x)
        # print("decoder K:", len(feats))
        # print("input x:", x.shape)
        if w_E2 is not None:
            y, weights = self.mixer(feats, w_E2, return_weights=True)
        else:
            y = sum(feats) / len(feats)
            weights = None
    
        y = self.act(self.norm(y))
    
        if return_weights:
            return y, weights
    
        return y
        
        # y = self.heavy(x, encoder_kernel_skip)
    
        #print("AFTER heavy, y:", None if y is None else y.shape)
    
        #assert y is not None, "heavy() returned None ❌"
    
        # y = self.act(self.norm(y))
    
    
    
    def heavy(self, x, encoder_kernel_skip):
        
        feats = self.conv_suite(x)
    
        #print("feats:", len(feats), feats[0].shape)
    
        B, C, D, H, W = feats[0].shape
        
        if isinstance(encoder_kernel_skip, torch.Tensor) and encoder_kernel_skip.dim() == 5:

            weights = encoder_kernel_skip
        
            weights = weights / (weights.std(dim=(2,3,4), keepdim=True) + 1e-6)
        
            weights = F.avg_pool3d(
                weights,
                kernel_size=(1, 3, 3),
                stride=1,
                padding=(0, 1, 1)
            )
        
            weights = F.interpolate(
                weights,
                size=x.shape[2:],
                mode="trilinear",
                align_corners=False
            )
        
            K = min(weights.shape[1], self.num_dec_kernels)
            weights = weights[:, :K]
            feats = feats[:K]
        
            weights = F.softmax(weights, dim=1)
        
        else:
            # 🔥 fallback: uniform weights
            K = min(len(feats), self.num_dec_kernels)
            feats = feats[:K]
        
            weights = torch.ones(
                (x.shape[0], K, x.shape[2], x.shape[3], x.shape[4]),
                device=x.device
            ) / K
    
        y = torch.zeros_like(feats[0])
    
        for i, f in enumerate(feats):
            # weights = encoder_kernel_skip
            # assert weights.shape[1] > i, f"weight mismatch at {i}"
            y = y + weights[:, i:i+1] * f
    
        #print("RETURNING y:", y.shape)
    
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
