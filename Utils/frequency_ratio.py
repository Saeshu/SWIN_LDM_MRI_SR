import torch

def frequency_ratio(x):
    # Handle all possible shapes

    if x.dim() == 5:  # (B, C, D, H, W)
        x = x[0]

    if x.dim() == 4:  # (C, D, H, W)
        x = x[0]

    if x.dim() == 3:  # (D, H, W)
        x = x[x.shape[0] // 2]  # take middle slice

    # Now MUST be (H, W)
    assert x.dim() == 2, f"Expected 2D image, got shape {x.shape}"

    img = x.float()

    # normalize
    img = img - img.mean()
    img = img / (img.std() + 1e-8)

    F_img = torch.fft.fftshift(torch.fft.fft2(img))
    mag = torch.abs(F_img)

    H, W = mag.shape  # now safe

    center_h, center_w = H // 2, W // 2

    Y, X = torch.meshgrid(
        torch.arange(H, device=img.device),
        torch.arange(W, device=img.device),
        indexing="ij"
    )

    dist = torch.sqrt((Y - center_h)**2 + (X - center_w)**2)

    low_mask = dist < (H // 6)
    high_mask = dist > (H // 3)

    low_energy = mag[low_mask].mean()
    high_energy = mag[high_mask].mean()

    return (low_energy / (high_energy + 1e-8)).item()
