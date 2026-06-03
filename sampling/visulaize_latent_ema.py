@torch.no_grad()
def visualize_with_ema(
    ema,
    ae,
    noise_sched,
    latent_shape,
    device,
    lr,
    hr,
    guidance_scale=1.5,
    title="EMA Sample"
):
    lr = lr.to(device)
    hr = hr.to(device)

    # -----------------------------
    # Upsample LR (correct way)
    # -----------------------------
    
    
    # -----------------------------
    # Encode conditioning
    # -----------------------------
    z_lr_out = ae.encode(lr)
    z_lr = z_lr_out[0] if isinstance(z_lr_out, (list, tuple)) else z_lr_out
    
    #z_lr_small = z_lr.mean(dim=1, keepdim=True)
    
    z_lr = F.interpolate(
        z_lr,
        size=latent_shape[2:],
        mode="trilinear",
        align_corners=False
    )
    
    cond = z_lr
    # -----------------------------
    # Sample residual → full latent
    # -----------------------------
    z = sample_latent_ema(
        ema=ema,
        noise_sched=noise_sched,
        shape=latent_shape,
        cond=cond,
        guidance_scale=guidance_scale,
        device=device
    )

    # -----------------------------
    # Decode to image
    # -----------------------------
    x = ae.decode(z)
    x = x.squeeze().cpu()

    if x.dim() == 4:
        x = x[0]

    D, H, W = x.shape

    slices = {
        "Axial": x[D // 2],
        "Coronal": x[:, H // 2, :],
        "Sagittal": x[:, :, W // 2],
    }

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(title)

    for ax, (name, img) in zip(axes, slices.items()):
        ax.imshow(img, cmap="gray")
        ax.set_title(name)
        ax.axis("off")

    score = structure_score(x)
    print("score:", score)

    ratio = frequency_ratio(x)
    print("frequency ratio:", ratio)

    plt.show()
