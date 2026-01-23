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
        self.down1 = SpatialDownsample3D()
        self.enc2 = AnisotropicSwinBlock(64, 128, depth_kernels=(3,5), use_attention=True)
        self.enc3 = AnisotropicSwinBlock(128, 256, depth_kernels=(3,5,7), use_attention=True)

        # ---- decoder ----
        self.dec2 = DecoderBlock(
        in_ch=256,
        out_ch=128,
        use_depth=True,
        enc_kernel_dim=ENC_KERNEL_DIM,
        upsample=True
        )
        
        # dec1: REDUCE paths
        self.dec1 = DecoderBlock(
            in_ch=128,
            out_ch=64,
            use_depth=False,     # 🔥 REMOVE depth path
            enc_kernel_dim=None,
            upsample=True
        )
        
        # dec0: LIGHTWEIGHT
        self.dec0 = DecoderBlock(
            in_ch=64,
            out_ch=32,
            use_depth=False,
            enc_kernel_dim=None,
            upsample=False
        )

        self.out = OutputRefinementHead(32, out_ch=1)

    def encode(self, x):
          x = self.enc0(x)
          x = self.down0(x)
          x = self.enc1(x)
          x = self.down1(x)
          x, _ = self.enc2(x, return_weights=True)
          x = self.enc3(x)
          return x

      # --- decoder only ---
      def decode(self, z, w_E2=None):
          # NOTE: for now, no skip — diffusion doesn't need it
          z = self.dec2(z, encoder_kernel_skip=w_E2, bias_strength=1.0)
          z = self.dec1(z)
          z = self.dec0(z)
          return self.out(z)

      def forward(self, x):
          z = self.encode(x)
          return self.decode(z)
