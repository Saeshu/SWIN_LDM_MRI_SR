import torch
import torch.nn.functional as F


def compute_losses(
    ae,
    x0_pred,
    z_res,
    z_hr,
    z_lr,
    v_pred,
    v_target,
    perc_net=None,
):
    """
    Compute all training losses.

    Returns
    -------
    dict
    """

    ####################################################
    # Diffusion loss
    ####################################################

    mse_loss = F.mse_loss(
        v_pred,
        v_target,
    )

    ####################################################
    # Residual loss
    ####################################################

    res_loss = F.l1_loss(
        x0_pred,
        z_res,
    )

    ####################################################
    # Reconstruction
    ####################################################

    recon = x0_pred + z_lr

    recon_loss = F.l1_loss(
        recon,
        z_hr,
    )

    ####################################################
    # Optional perceptual
    ####################################################

    perc_loss = torch.tensor(
        0.0,
        device=x0_pred.device,
    )

    if perc_net is not None:

        x_base = x0_pred.mean(
            dim=1,
            keepdim=True,
        )

        z_base = z_res.mean(
            dim=1,
            keepdim=True,
        )

        x_input = torch.cat(

            [
                F.avg_pool3d(x_base, 2),
                x_base - F.interpolate(
                    F.avg_pool3d(
                        F.avg_pool3d(x_base, 2),
                        2,
                    ),
                    size=F.avg_pool3d(
                        x_base,
                        2,
                    ).shape[2:],
                    mode="trilinear",
                    align_corners=False,
                ),
            ],

            dim=1,

        )

        z_input = torch.cat(

            [
                F.avg_pool3d(z_base, 2),
                z_base - F.interpolate(
                    F.avg_pool3d(
                        F.avg_pool3d(z_base, 2),
                        2,
                    ),
                    size=F.avg_pool3d(
                        z_base,
                        2,
                    ).shape[2:],
                    mode="trilinear",
                    align_corners=False,
                ),
            ],

            dim=1,

        )

        f_pred, _ = perc_net(
            x_input
        )

        with torch.no_grad():

            f_gt, _ = perc_net(
                z_input
            )

        perc_loss = F.l1_loss(
            f_pred,
            f_gt,
        )

    ####################################################
    # Total
    ####################################################

    total = (

        1.0 * mse_loss

        +

        0.5 * res_loss

        +

        0.5 * recon_loss

        +

        0.05 * perc_loss

    )

    ####################################################
    # Return
    ####################################################

    return {

        "total": total,

        "mse": mse_loss,

        "recon": recon_loss,

        "res": res_loss,

        "perc": perc_loss,

    }
