#need to add utils
import torch
import os
import math

def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path):
    return torch.load(path, map_location="cpu")

def timestep_embedding(timesteps, dim):
    """
    Create sinusoidal timestep embeddings.

    Args:
        timesteps: Long tensor of shape [B]
        dim: embedding dimension

    Returns:
        Tensor of shape [B, dim]
    """
    half = dim // 2
    device = timesteps.device

    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=device) / half
    )

    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))

    return emb
