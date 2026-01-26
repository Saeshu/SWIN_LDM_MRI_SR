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


def match_shape(x, ref):
    """
    Match x spatial shape to ref by symmetric padding or cropping.
    """
    _, _, D, H, W = ref.shape
    d, h, w = x.shape[2:]

    pd = D - d
    ph = H - h
    pw = W - w

    # Pad if too small
    if pd > 0 or ph > 0 or pw > 0:
        x = F.pad(
            x,
            (
                pw // 2, pw - pw // 2,
                ph // 2, ph - ph // 2,
                pd // 2, pd - pd // 2,
            )
        )

    # Crop if too large
    x = x[:, :, :D, :H, :W]
    return x

def random_hw_patch_3d(vol, patch_hw=(192, 192)):
    """
    vol: [1, D, H, W]  (single-channel MRI)
    patch_hw: (ph, pw)

    returns: [1, D, ph, pw]
    """
    _, D, H, W = vol.shape
    ph, pw = patch_hw

    assert H >= ph and W >= pw, "Patch size too big for H/W"

    h0 = random.randint(0, H - ph)
    w0 = random.randint(0, W - pw)

    patch = vol[:, :, h0:h0+ph, w0:w0+pw]
    return patch

