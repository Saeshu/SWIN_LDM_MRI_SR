# models/ae.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ShapedEncoder3D import AnisotropicSwinBlock, SpatialDownsample3D
from .Decoder import DecoderBlock, OutputRefinementHead
ENC_KERNEL_DIM = 4
class WE2FiLM(nn.Module):
    def __init__(self, hidden_ch=64):
        super().__init__()

        self.hidden_ch = hidden_ch

        self.proj = None
        self.to_scale = None
        self.to_shift = None
        self.scale_proj = None
        self.shift_proj = None

    def forward(self, h, w_e2):
        """
        h:    [B, C, D, H, W]
        w_e2: [B, K, D, H, W]
        """

        # -----------------------------
        # Resize
        # -----------------------------
        w = F.interpolate(
            w_e2,
            size=h.shape[2:],
            mode="trilinear",
            align_corners=False
        )

        # -----------------------------
        # Normalize (VERY IMPORTANT)
        # -----------------------------
        std = w.std(dim=(2,3,4), keepdim=True)
        std = torch.clamp(std, min=1e-3)
        w = w / std

        # -----------------------------
        # Lazy init
        # -----------------------------
        if self.proj is None:
            in_ch = w.shape[1]      # K
            out_ch = h.shape[1]     # feature channels

            self.proj = nn.Sequential(
                nn.Conv3d(in_ch, self.hidden_ch, 1),
                nn.SiLU(),
                nn.Conv3d(self.hidden_ch, self.hidden_ch, 3, padding=1),
                nn.SiLU()
            ).to(h.device)

            self.to_scale = nn.Conv3d(self.hidden_ch, self.hidden_ch, 1).to(h.device)
            self.to_shift = nn.Conv3d(self.hidden_ch, self.hidden_ch, 1).to(h.device)

            self.scale_proj = nn.Conv3d(self.hidden_ch, out_ch, 1).to(h.device)
            self.shift_proj = nn.Conv3d(self.hidden_ch, out_ch, 1).to(h.device)

        # -----------------------------
        # Feature extraction
        # -----------------------------
        w_feat = self.proj(w)

        # -----------------------------
        # Generate modulation
        # -----------------------------
        scale = torch.tanh(self.to_scale(w_feat))
        shift = self.to_shift(w_feat)

        scale = self.scale_proj(scale)
        shift = self.shift_proj(shift)

        # -----------------------------
        # 🔥 FiLM (WEAK)
        # -----------------------------
        h = h * (1 + 0.01 * scale) + 0.01 * shift

        return h
        
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.we2 = WE2FiLM()
        # ---- encoder ----
        self.enc0 = AnisotropicSwinBlock(1, 32, depth_kernels=(), use_attention=False)
        self.down0 = SpatialDownsample3D()

        self.enc1 = AnisotropicSwinBlock(32, 64, depth_kernels=(3,), use_attention=False)
        
        self.down1 = SpatialDownsample3D()  # 🔥 removed

        self.enc2 = AnisotropicSwinBlock(64, 128, depth_kernels=(3,5), use_attention=False)
        
        self.enc3 = AnisotropicSwinBlock(128, 256, depth_kernels=(3,), use_attention=False)

        # 🔥 NEW: projection instead of mean
        self.latent_proj = nn.Conv3d(256, 4, kernel_size=1)

        # ---- decoder ----
        self.dec2 = DecoderBlock(256, 128, use_depth=True, enc_kernel_dim=ENC_KERNEL_DIM, upsample=True)
        self.dec1 = DecoderBlock(128, 64, use_depth=False, enc_kernel_dim=None, upsample=True)
        self.dec0 = DecoderBlock(64, 32, use_depth=False, enc_kernel_dim=None, upsample=False)

        self.out = OutputRefinementHead(32, out_ch=1)

    def encode(self, x):
        x = self.enc0(x)
        x = self.down0(x)
    
        x = self.enc1(x)
        x = self.down1(x)

        out = self.enc2(x, return_weights=True)
        if isinstance(out, tuple):
            x, w_E2 = out
        else:
            x = out
            w_E2 = None
            
        x = self.enc3(x)
    
        # 🔥 TEMP FIX: make compatible with decoder
       
    
        return x, w_E2

    def decode(self, z, w_E2=None, mode="baseline"):
        """
        mode:
            "baseline" → no conditioning
            "we2"      → kernel bias conditioning
            "film"     → FiLM conditioning
        """
    
        # -------------------------
        # Stage 1
        # -------------------------
        if mode == "we2" and w_E2 is not None:
            z = self.dec2(z, encoder_kernel_skip=w_E2)
        else:
            z = self.dec2(z, encoder_kernel_skip=None)
    
        # -------------------------
        # FiLM injection
        # -------------------------
        if mode == "film" and w_E2 is not None:
            z = self.we2_film(z, w_E2)
    
        # -------------------------
        # Remaining decoder
        # -------------------------
        z = self.dec1(z)
        z = self.dec0(z)
    
        return self.out(z)
    
    def forward(self, x, mode="baseline"):
        """
        mode:
            "baseline" → no conditioning
            "we2"      → kernel bias conditioning
            "film"     → FiLM conditioning
        """
    
        z, w = self.encode(x)
    
        if mode == "baseline":
            return self.decode(z, None, mode="baseline")
    
        elif mode == "we2":
            return self.decode(z, w, mode="we2")
    
        elif mode == "film":
            return self.decode(z, w, mode="film")
    
        else:
            raise ValueError(f"Unknown mode: {mode}")
