import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

class SpatialKernelMixer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feats, w):
        """
        feats: list of K tensors [B, C, D, H, W]
        w:     [B, K, D, H, W]
        """

        K = len(feats)

        # -----------------------------
        # Resize weights
        # -----------------------------
        w = F.interpolate(
            w,
            size=feats[0].shape[2:],
            mode="trilinear",
            align_corners=False
        )

        # -----------------------------
        # Match kernel count
        # -----------------------------
        if w.shape[1] != K:
            K_min = min(K, w.shape[1])
            feats = feats[:K_min]
            w = w[:, :K_min]
            K = K_min

        # -----------------------------
        # Normalize across kernels
        # -----------------------------
        w = w / (w.std(dim=1, keepdim=True) + 1e-6)
        weights = F.softmax(w, dim=1)   # [B,K,D,H,W]

        # -----------------------------
        # Spatial mixing
        # -----------------------------
        y = 0
        for i, f in enumerate(feats):
            y = y + weights[:, i:i+1] * f

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
    """
    Anisotropic convolution suite for reconstruction.
    Spatial kernels dominate.
    """

    def __init__(self, in_ch, out_ch, use_depth=True):
        super().__init__()

        # Spatial refinement
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
            # Short-range depth regularization only
            self.conv_1x1x3 = nn.Conv3d(
                in_ch, out_ch, kernel_size=(3, 1, 1), padding=(1, 0, 0)
            )

        # Channel mixer
        self.conv_1x1x1 = nn.Conv3d(in_ch, out_ch, kernel_size=1)

        self.num_paths = 4

    def forward(self, x):
        feats = [
            self.conv_3x3x1(x),
            self.conv_1x3x1(x),
            self.conv_3x1x1(x),
            self.conv_1x1x1(x)
        ]

        # if self.use_depth:
        #     feats.append(self.conv_1x1x3(x))

        #feats.append(self.conv_1x1x1(x))

        return feats


# --------------------------------------------------
# Decoder block (upsample + conv suite + kernel mixing)
# --------------------------------------------------

class DecoderBlock(nn.Module):
    """
    One decoder stage:
    - spatial upsampling (H, W only)
    - anisotropic reconstruction convs
    - kernel mixing (optionally biased by encoder intent)
    """

    def __init__(self, in_ch, out_ch, use_depth=True, enc_kernel_dim=None, upsample=True):
        super().__init__()

        self.upsample_enabled = upsample
        self.upsample = SpatialUpsample3D(scale_factor=2)
        
        self.conv_suite = DecoderConvSuite(
            in_ch=in_ch,
            out_ch=out_ch,
            use_depth=use_depth
        )
        #self.conv_suite.num_paths = 4 + (1 if use_depth else 0)
        self.num_dec_kernels = self.conv_suite.num_paths

        # Decoder-side kernel logits (learned)
        self.logits = nn.Parameter(torch.zeros(self.num_dec_kernels))

        # 🔑 Learned projection: encoder intent → decoder kernel space
        if enc_kernel_dim is not None:
            self.enc_to_dec = nn.Linear(enc_kernel_dim, self.num_dec_kernels)
        else:
            self.enc_to_dec = None

        self.norm = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()

    def forward(self, x, encoder_kernel_skip=None, bias_strength=0.01):

        if self.upsample_enabled:
            x = self.upsample(x)
    
        feats = self.conv_suite(x)
    
        # -----------------------------
        # 🔥 BASELINE weights (learned)
        # -----------------------------
        weights = torch.softmax(self.logits, dim=0)   # [K]
    
        # -----------------------------
        # 🔥 OPTIONAL w_E2 bias (SAFE)
        # -----------------------------
        if encoder_kernel_skip is not None:
            w = encoder_kernel_skip
    
            # resize to match
            w = F.interpolate(
                w,
                size=x.shape[2:],
                mode="trilinear",
                align_corners=False
            )
    
            # normalize (critical)
            w = w / (w.std(dim=(2,3,4), keepdim=True) + 1e-6)
    
            # reduce spatial → global signal
            w_global = w.mean(dim=(2,3,4))   # [B, K]
    
            # average across batch → stable
            w_global = w_global.mean(dim=0)  # [K]
    
            # convert to soft bias
            w_bias = torch.softmax(w_global, dim=0)
            # print(w_bias.shape)
            # print(weights.shape)
    
            # 🔥 blend with learned weights (VERY WEAK)
            weights = (1 - bias_strength) * weights + bias_strength * w_bias
    
        # -----------------------------
        # 🔥 feature mixing
        # -----------------------------
        y = 0
        for i, f in enumerate(feats):
            y = y + weights[i] * f
    
        y = self.act(self.norm(y))
    
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
            assert weights.shape[1] > i, f"weight mismatch at {i}"
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
