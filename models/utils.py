#need to add utils
import torch
import os
import math
import random
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




def random_patch_3d(vol, patch_size=64):
    """
    vol: [1, D, H, W]  (single-channel MRI)
    returns: [1, patch, patch, patch]
    """
    _, D, H, W = vol.shape
    ps = patch_size

    assert D >= ps and H >= ps and W >= ps, "Patch size too big"

    d0 = random.randint(0, D - ps)
    h0 = random.randint(0, H - ps)
    w0 = random.randint(0, W - ps)

    patch = vol[:, d0:d0+ps, h0:h0+ps, w0:w0+ps]
    return patch

