# Diffusion/validate.py

import torch
import torch.nn.functional as F

from Diffusion.losses import (
    sample_timesteps,
    get_alpha_bar,
    add_noise,
    make_v_target,
    predict_x0,
    diffusion_loss,
    prediction_statistics,
)


# ---------------------------------------------------------
# Image metrics
# ---------------------------------------------------------

def compute_psnr(pred, target):

    mse = F.mse_loss(pred, target)

    if mse.item() == 0:
        return 99.0

    return (-10 * torch.log10(mse)).item()


def laplacian(x):

    return x - F.avg_pool3d(
        x,
        kernel_size=3,
        stride=1,
        padding=1,
    )


def high_frequency_loss(pred, target):

    return F.l1_loss(
        laplacian(pred),
        laplacian(target)
    )


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

@torch.no_grad()
def validate(

    ae,
    unet,
    loader,
    noise_scheduler,
    device,

    stage=1,
    alpha=1.0,

):

    ae.eval()
    unet.eval()

    #########################################################
    # Running statistics
    #########################################################

    stats = {

        "loss": 0.0,
        "mse": 0.0,

        "corr": 0.0,
        "cosine": 0.0,

        "pred_mean": 0.0,
        "pred_std": 0.0,

        "target_mean": 0.0,
        "target_std": 0.0,

        "residual_l1": 0.0,
        "residual_mse": 0.0,

        "psnr": 0.0,
        "hf_loss": 0.0,

    }

    batches = 0

    #########################################################
    # Cached tensors
    #########################################################

    cache = None

    #########################################################
    # Loop
    #########################################################

    for hr, lr in loader:

        hr = hr.to(device)
        lr = lr.to(device)

        #####################################################
        # Encode
        #####################################################

        out_hr = ae.encode(hr)
        out_lr = ae.encode(lr)

        if isinstance(out_hr, tuple):
            z_hr, w = out_hr
        else:
            z_hr = out_hr
            w = None

        if isinstance(out_lr, tuple):
            z_lr = out_lr[0]
        else:
            z_lr = out_lr

        #####################################################
        # Residual latent
        #####################################################

        z_res = z_hr - z_lr

        #####################################################
        # Diffusion
        #####################################################

        t = sample_timesteps(

            batch_size=z_res.shape[0],
            T=noise_scheduler.num_timesteps,
            device=device,

            curriculum="quadratic",

            max_fraction=1.0,

        )

        z_noisy, noise = add_noise(
            z_res,
            noise_scheduler,
            t,
        )

        alpha_bar = get_alpha_bar(
            noise_scheduler,
            t,
        )

        v_target = make_v_target(
            z_res,
            noise,
            alpha_bar,
        )

        #####################################################
        # UNet
        #####################################################

        v_pred = unet(

            z_noisy,

            t,

            cond=z_lr,

            w_e2=w,

            alpha=alpha,

        )

        #####################################################
        # Recover residual
        #####################################################

        x0_pred = predict_x0(

            z_noisy,

            v_pred,

            alpha_bar,

        )

        #####################################################
        # Loss
        #####################################################

        loss = diffusion_loss(
            v_pred,
            v_target,
        )

        #####################################################
        # Prediction statistics
        #####################################################

        pred_stats = prediction_statistics(
            v_pred,
            v_target,
        )

        #####################################################
        # Residual statistics
        #####################################################

        residual_l1 = F.l1_loss(
            x0_pred,
            z_res,
        )

        residual_mse = F.mse_loss(
            x0_pred,
            z_res,
        )

        #####################################################
        # Decode
        #####################################################

        z_final = z_lr + x0_pred

        recon = ae.decode(
            z_final,
            w,
        )

        #####################################################
        # Image metrics
        #####################################################

        psnr = compute_psnr(
            recon,
            hr,
        )

        hf = high_frequency_loss(
            recon,
            hr,
        )

        #####################################################
        # Accumulate
        #####################################################

        stats["loss"] += loss.item()

        stats["mse"] += pred_stats["mse"]

        stats["corr"] += pred_stats["corr"]

        stats["cosine"] += pred_stats["cosine"]

        stats["pred_mean"] += pred_stats["pred_mean"]
        stats["pred_std"] += pred_stats["pred_std"]

        stats["target_mean"] += pred_stats["target_mean"]
        stats["target_std"] += pred_stats["target_std"]

        stats["residual_l1"] += residual_l1.item()
        stats["residual_mse"] += residual_mse.item()

        stats["psnr"] += psnr
        stats["hf_loss"] += hf.item()

        #####################################################
        # Save one batch for visualization
        #####################################################

        if cache is None:

            cache = {

                "hr": hr[:1].cpu(),

                "lr": lr[:1].cpu(),

                "recon": recon[:1].cpu(),

                "z_hr": z_hr[:1].cpu(),

                "z_lr": z_lr[:1].cpu(),

                "z_res": z_res[:1].cpu(),

                "x0_pred": x0_pred[:1].cpu(),

                "v_pred": v_pred[:1].cpu(),

                "v_target": v_target[:1].cpu(),

                "t": t[:1].cpu(),

            }

        batches += 1

    #########################################################
    # Average
    #########################################################
    
    for k in stats:

        stats[k] /= batches

    diagnostics = {

    "z_hr": z_hr
    "z_lr": z_lr,
    "z_res": z_res,
    "x0_pred": x0_pred,
    "v_pred": v_pred,
    "v_target": v,
    "routing": w,
}

    return stats, cache, diagnostics
