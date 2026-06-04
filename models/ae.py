# models/ae.py
import torch
import torch.nn as nn

from .ShapedEncoder3D import AnisotropicSwinBlock, SpatialDownsample3D
from .Decoder import DecoderBlock, OutputRefinementHead
ENC_KERNEL_DIM = 4 
# models/ae.py
import torch
import torch.nn as nn

from .ShapedEncoder3D import AnisotropicSwinBlock, SpatialDownsample3D
from .Decoder import DecoderBlock, OutputRefinementHead
ENC_KERNEL_DIM = 4 
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # ---- encoder ----
        self.enc0 = AnisotropicSwinBlock(1, 32, depth_kernels=(), use_attention=False)
        self.down0 = SpatialDownsample3D()

        self.enc1 = AnisotropicSwinBlock(32, 64, depth_kernels=(3,), use_attention=False)
        
        self.down1 = nn.Identity()  # 🔥 removed

        self.enc2 = AnisotropicSwinBlock(64, 128, depth_kernels=(3,5), use_attention=True)
        self.enc3 = AnisotropicSwinBlock(128, 256, depth_kernels=(3,5,7), use_attention=True)

        # 🔥 NEW: projection instead of mean
        self.latent_proj = nn.Conv3d(256, 1, kernel_size=1)

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
    
        x, w_E2 = self.enc2(x, return_weights=True)
        x = self.enc3(x)
    
        # 🔥 TEMP FIX: make compatible with decoder
        
        return x, w_E2

    def decode(self, z, w_E2=None):
        z = self.dec2(z, encoder_kernel_skip=w_E2, bias_strength=1.0)
        z = self.dec1(z)
        z = self.dec0(z)
        return self.out(z)

    def forward(self, x):
        z, w = self.encode(x)
        return self.decode(z, w)
          z = self.encode(x)
          return self.decode(z)
