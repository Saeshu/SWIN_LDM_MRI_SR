import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.utils import timestep_embedding
#from Diffusion.convsuite import TimeGatedConvSuite
from Diffusion.schedule import SinusoidalTimeEmbedding
from Diffusion.LinearNoise import NoiseScheduler

class SpatialSuite(nn.Module):
  def __init__(self, c):
      super().__init__()
      self.net = nn.Sequential(
          nn.Conv3d(c, c, kernel_size=(3,3,1), padding=(1,1,0)),
          nn.SiLU(),
          nn.Conv3d(c, c, kernel_size=(1,3,1), padding=(0,1,0)),
          nn.SiLU(),
          nn.Conv3d(c, c, kernel_size=(3,1,1), padding=(1,0,0)),
      )

  def forward(self, x):
      return self.net(x)

class MidSliceSuite(nn.Module):
  def __init__(self, c):
      super().__init__()
      self.net = nn.Sequential(
          nn.Conv3d(c, c, kernel_size=(1,1,3), padding=(0,0,1)),
          nn.SiLU(),
          nn.Conv3d(c, c, kernel_size=(1,1,5), padding=(0,0,2)),
      )

  def forward(self, x):
      return self.net(x)

class LongSliceSuite(nn.Module):
  def __init__(self, c):
      super().__init__()
      self.net = nn.Sequential(
          nn.Conv3d(c, c, kernel_size=(1,1,7), padding=(0,0,3)),
          nn.SiLU(),
          nn.Conv3d(c, c, kernel_size=(1,1,9), padding=(0,0,4)),
      )
  def forward(self, x):
      return self.net(x)





class TimeGatedConvSuite(nn.Module):
  def __init__(self, channels, time_dim=128):
      super().__init__()

      self.time_embed = SinusoidalTimeEmbedding(time_dim)
      self.time_mlp = nn.Sequential(
          nn.Linear(time_dim, time_dim),
          nn.SiLU(),
          nn.Linear(time_dim, 3)  # gates for A, B, C
      )

      self.spatial = SpatialSuite(channels)
      self.mid = MidSliceSuite(channels)
      self.long = LongSliceSuite(channels)

  def forward(
      self,
      x,
      t,
      expert_mask=None,
      return_gates=False,
):
      """
      Parameters
      ----------
      x : [B,C,D,H,W]
  
      t : [B]
  
      expert_mask :
          None      -> normal routing
  
          [1,1,1]   -> normal
  
          [1,0,1]   -> disable mid
  
          [0,1,0]   -> only mid
  
      return_gates :
          return routing probabilities
      """

    # -------------------------------------------------
    # Compute routing
    # -------------------------------------------------

      te = self.time_embed(t)

      logits = self.time_mlp(te)

      gates = torch.softmax(logits, dim=-1)

    
    # -------------------------------------------------
    # Optional masking
    # -------------------------------------------------

      if expert_mask is not None:

          mask = torch.tensor(
              expert_mask,
              device=gates.device,
              dtype=gates.dtype,
          )

          gates = gates * mask

          # Renormalize

          gates = gates / (
              gates.sum(
                  dim=-1,
                  keepdim=True
              ) + 1e-8
          )

      # -------------------------------------------------
      # Experts
      # -------------------------------------------------

      spatial = self.spatial(x)

      mid = self.mid(x)

      long = self.long(x)

      # -------------------------------------------------
      # Broadcast gates
      # -------------------------------------------------

      gates = gates[..., None, None, None, None]

      out = (

          gates[:,0] * spatial +

          gates[:,1] * mid +

          gates[:,2] * long

      )

      # -------------------------------------------------
      # Return
      # -------------------------------------------------

      if return_gates:

          return out, gates.squeeze(-1).squeeze(-1).squeeze(-1).squeeze(-1)

      return out

    

  class BottleneckBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.conv = nn.Conv3d(channels, channels, 3, padding=1)
        self.temporal_suite = TimeGatedConvSuite(channels)

    def forward(self, x, t):
        h = self.norm(x)
        h = F.silu(h)
        h = self.conv(h)

        # temporal / slice-aware correction
        h = h + self.temporal_suite(h, t)

        return h

