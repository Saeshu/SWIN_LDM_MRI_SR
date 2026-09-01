# models/ae.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .ShapedEncoder3D import AnisotropicSwinBlock, SpatialDownsample3D
from .Decoder import DecoderBlock, OutputRefinementHead
ENC_KERNEL_DIM = 5
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
        # -----------------------------
        # Normalize
        # -----------------------------
        std = torch.clamp(w.std(dim=(2,3,4), keepdim=True), min=1e-3)
        w = w / std
        
        # -----------------------------
        # Threshold (detached)
        # -----------------------------
        thr = (w.abs().mean() + 0.5 * w.abs().std()).detach()
        
        # -----------------------------
        # Mask
        # -----------------------------
        mask = torch.sigmoid(20 * ((w.abs() - thr) / (w.abs().std() + 1e-6)))
        
        # 🔥 collapse kernel dimension
        mask = mask.mean(dim=1, keepdim=True)   # [B,1,D,H,W]
        
        # -----------------------------
        # FiLM (WEAK + MASKED)
        # -----------------------------
        h = h * (1 + 0.001 * mask * scale) + 0.001 * mask * shift

        return h
        
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.we2_film = WE2FiLM()
        # ---- encoder ----
        self.enc0 = AnisotropicSwinBlock(1, 32, use_attention=False)
        self.down0 = SpatialDownsample3D()
        
        self.enc1 = AnisotropicSwinBlock(32, 64, use_attention=False)
        self.down1 = SpatialDownsample3D()
        
        self.enc2 = AnisotropicSwinBlock(64, 128, use_attention=True, shift=False)
        
        self.enc3 = AnisotropicSwinBlock(128, 256, use_attention=True, shift = False)

        # 🔥 NEW: projection instead of mean
        self.latent_proj = nn.Conv3d(256, 4, kernel_size=1)

        # ---- decoder ----
        self.dec2 = DecoderBlock(
            256,
            128,
            upsample=True,
            use_routing=True,
            spatial_reduction=False,
            channel_reduction=True
        )
        
        self.dec1 = DecoderBlock(
            128,
            64,
            upsample=True,
            use_routing=False,
            spatial_reduction=False,
            channel_reduction=True
        )
        
        self.dec0 = DecoderBlock(
            64,
            32,
            upsample=False,
            use_routing=False,
            spatial_reduction=False,
            channel_reduction=True
        )
        self.out = OutputRefinementHead(32, out_ch=1)

    def encode(self, x, return_features=False):

        features = {}
    
        # --------------------------------------------------
        # Encoder block 0
        # --------------------------------------------------
    
        x = self.enc0(x)
    
        if return_features:
            features["enc0"] = x
    
        x = self.down0(x)
    
        # --------------------------------------------------
        # Encoder block 1
        # --------------------------------------------------
    
        x = self.enc1(x)
    
        if return_features:
            features["enc1"] = x
    
        x = self.down1(x)
    
        # --------------------------------------------------
        # Encoder block 2
        # --------------------------------------------------
    
        out = self.enc2(
            x,
            return_weights=True,
        )
    
        if isinstance(out, tuple):
            x, w_E2 = out
        else:
            x = out
            w_E2 = None
    
        if return_features:
            features["enc2"] = x
    
        # --------------------------------------------------
        # Encoder block 3
        # --------------------------------------------------
    
        x = self.enc3(x)
    
        if return_features:
            features["enc3"] = x
    
        # --------------------------------------------------
        # Return
        # --------------------------------------------------
    
        if return_features:
            return {
                "features": features,
                "latent": x,
                "w_E2": w_E2,
            }

        return x, w_E2
    
    def decode(
    self,
    z,
    return_weights=False,
):

        if return_weights:
    
            z, weights = self.dec2(
                z,
                return_weights=True,
            )

    
        else:
    
            z = checkpoint(
            self.dec2,
            z,
            use_reentrant=False,
        )
    
            weights = None
    
        # ------------------------------------------------
        # Remaining decoder
        # ------------------------------------------------
    
        z = checkpoint(
            self.dec1,
            z,
            use_reentrant=False,
        )
    
        z = checkpoint(
            self.dec0,
            z,
            use_reentrant=False,
        )
    
        out = self.out(z)
    
        if return_weights:
    
            return out, weights
    
        return out
        
    def forward(self, x, return_weights=False):

        z, w_enc = self.encode(x)
    
      

        if return_weights:
    
            out, w_dec = self.decode(
                z,
                return_weights=True
            )
    
            return out, w_enc, w_dec
    
        out = self.decode(z)
    
        return out
