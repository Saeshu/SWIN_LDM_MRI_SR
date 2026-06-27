@torch.no_grad()
def run_callbacks(
    self,
    epoch,
    hr,
    lr,
):
    """
    Execute registered callbacks.

    Parameters
    ----------
    epoch : int

    hr : [B,1,D,H,W]

    lr : [B,1,D,H,W]
    """

    ##########################################################
    # Nothing to do
    ##########################################################

    if all(cb is None for cb in self.callbacks.values()):
        return

    ##########################################################
    # Prepare a single sample
    ##########################################################

    hr = hr[:1].to(self.device)
    lr = lr[:1].to(self.device)

    ##########################################################
    # Encode once
    ##########################################################

    z_hr, w_e2 = self.ae.encode(hr)
    z_lr, _ = self.ae.encode(lr)

    residual_gt = z_hr - z_lr

    ##########################################################
    # Initial noisy latent
    ##########################################################

    z_init = torch.randn_like(residual_gt)

    ##########################################################
    # Common callback state
    ##########################################################

    callback_state = {

        "epoch": epoch,

        "ae": self.ae,

        "unet": self.ema.ema_model if self.ema is not None else self.unet,

        "scheduler": self.scheduler,

        "ema": self.ema,

        "logger": self.logger,

        "device": self.device,

        "hr": hr,

        "lr": lr,

        "z_hr": z_hr,

        "z_lr": z_lr,

        "z_init": z_init,

        "residual_gt": residual_gt,

        "w_e2": w_e2,

    }

    ##########################################################
    # Execute callbacks
    ##########################################################

    for name, callback in self.callbacks.items():

        if callback is None:
            continue

        freq = self.callback_frequency.get(name, 1)

        if epoch % freq != 0:
            continue

        ######################################################
        # Run callback
        ######################################################

        results = callback(**callback_state)

        ######################################################
        # Update logger
        ######################################################

        if results is None:
            continue

        if name == "trajectory":

            self.logger.update_trajectory(results)

        elif name == "moe":

            self.logger.update_moe(results)

        elif name == "validation":

            self.logger.update_validation(results)

        ######################################################
        # Future callbacks simply return results
        ######################################################
