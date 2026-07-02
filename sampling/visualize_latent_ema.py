import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sampling.sample_latent_ema import sample_latent_ema
from Utils.utils import structure_score
from Utils.utils import frequency_ratio


@torch.no_grad()
def visualize_with_ema(
    ema,
    ae,
    noise_sched,
    device,
    lr,
    hr,
    guidance_scale=1.5,
    title="EMA Sample",
):
    """
    Sample a super-resolved MRI using the EMA diffusion model
    and visualize orthogonal slices.
    """

    ############################################################
    # Move to device
    ############################################################

    ae.eval()
    ema.ema_model.eval()

    lr = lr.to(device)
    hr = hr.to(device)

    ############################################################
    # Encode LR
    ############################################################
    z_hr, _ = ae.encode(hr)
    z_lr, w_e2 = ae.encode(lr)
    z_lr = F.interpolate(
    z_lr,
    size=z_hr.shape[2:],
    mode="trilinear",
    align_corners=False,
    )
    ############################################################
    # Sample latent
    ############################################################

    sample = sample_latent_ema(

        ema=ema,

        noise_sched=noise_sched,

        cond=z_lr,

        w_e2 = w_e2,

        device=device,

        guidance_scale=guidance_scale,

        debug=False,

    )
    print("w_e2 shape :", w_e2.shape)
    print("Cond shape :", z_lr.shape)
    z_sr = sample["latent"]

    ############################################################
    # Decode
    ############################################################

    x = ae.decode(z_sr)

    x = x.squeeze().detach().cpu()

    ############################################################
    # Optional GT
    ############################################################

    gt = hr.squeeze().cpu()

    ############################################################
    # Central slices
    ############################################################

    D, H, W = x.shape

    pred_slices = {

        "Axial": x[D // 2],

        "Coronal": x[:, H // 2, :],

        "Sagittal": x[:, :, W // 2],

    }

    gt_slices = {

        "Axial": gt[D // 2],

        "Coronal": gt[:, H // 2, :],

        "Sagittal": gt[:, :, W // 2],

    }
    assert x.shape == gt.shape, (
    f"Prediction {x.shape}, GT {gt.shape}"
    )
    ############################################################
    # Plot
    ############################################################

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    fig.suptitle(title)

    for i, name in enumerate(pred_slices.keys()):

        axes[0, i].imshow(

            gt_slices[name],

            cmap="gray",

        )

        axes[0, i].set_title(f"GT {name}")

        axes[0, i].axis("off")

        axes[1, i].imshow(

            pred_slices[name],

            cmap="gray",

        )

        axes[1, i].set_title(f"Pred {name}")

        axes[1, i].axis("off")

    plt.tight_layout()

    plt.show()

    ############################################################
    # Diagnostics
    ############################################################

    print("=" * 60)

    print("Sampling diagnostics")

    print("=" * 60)

    print("Structure score :", structure_score(x))

    print("Frequency ratio :", frequency_ratio(x))

    print("Latent std      :", z_sr.std().item())

    print("Latent mean     :", z_sr.mean().item())

    print("Residual std    :", sample["residual"].std().item())

    print("=" * 60)

    return {

        "prediction": x,

        "ground_truth": gt,

        "latent": z_sr,

        "residual": sample["residual"],

        "contractions": sample["contractions"],

    }
