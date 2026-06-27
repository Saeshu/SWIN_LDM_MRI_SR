def train_step(
    self,
    hr: torch.Tensor,
    lr: torch.Tensor,
    debug: bool = False,
):
    """
    Perform one diffusion optimization step.

    Parameters
    ----------
    hr : [B,1,D,H,W]

    lr : [B,1,D,H,W]

    Returns
    -------
    dict
    """

    ########################################################
    # Device
    ########################################################

    hr = hr.to(self.device)
    lr = lr.to(self.device)

    ########################################################
    # Encode
    ########################################################

    with torch.no_grad():

        encoded = self._encode(hr, lr)

    ########################################################
    # Forward diffusion
    ########################################################

    outputs = self._forward(encoded)

    ########################################################
    # Loss
    ########################################################

    losses = self._compute_loss(outputs)

    ########################################################
    # Optimizer
    ########################################################

    self._optimizer_step(losses["total"])

    ########################################################
    # Return
    ########################################################

    result = {

        "loss": losses["total"].detach(),

        "losses": losses,

        "timestep": outputs["t"],

        "v_pred": outputs["v_pred"].detach(),

        "v_target": outputs["v_target"].detach(),

        "x0_pred": outputs["x0_pred"].detach(),

        "z_res": outputs["z_res"].detach(),

        "z_lr": outputs["z_lr"].detach(),

    }

    if debug:

        result.update({

            "z_hr": outputs["z_hr"].detach(),

            "z_noisy": outputs["z_noisy"].detach(),

            "noise": outputs["noise"].detach(),

            "alpha_bar": outputs["alpha_bar"].detach(),

        })

    return result
