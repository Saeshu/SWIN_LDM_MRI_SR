from tqdm import tqdm
import torch


def train_epoch(
    self,
    dataloader,
):
    """
    Train for one epoch.

    Parameters
    ----------
    dataloader

    Returns
    -------
    dict
        Epoch statistics.
    """

    ##########################################################
    # Train mode
    ##########################################################

    self.unet.train()

    if hasattr(self, "adapter"):
        self.adapter.train()

    ##########################################################
    # Reset logger
    ##########################################################

    self.logger.running.clear()

    ##########################################################
    # Optimizer
    ##########################################################

    self.optimizer.zero_grad(set_to_none=True)

    ##########################################################
    # Progress bar
    ##########################################################

    pbar = tqdm(

        enumerate(dataloader),

        total=len(dataloader),

        leave=False,

        desc="Training",

    )

    ##########################################################
    # Loop
    ##########################################################

    for step, batch in pbar:

        ######################################################
        # Load batch
        ######################################################

        if isinstance(batch, (list, tuple)):

            hr, lr = batch

        else:

            raise ValueError(
                "Expected dataloader to return (hr, lr)"
            )

        ######################################################
        # Forward
        ######################################################

        outputs = self.train_step(

            hr,

            lr,

            debug=False,

        )

        ######################################################
        # Loss
        ######################################################

        loss = outputs["loss"]

        ######################################################
        # Backward
        ######################################################

        loss = loss / self.accum_steps

        self.scaler.scale(loss).backward()

        ######################################################
        # Optimizer step
        ######################################################

        if (

            (step + 1) % self.accum_steps == 0

            or

            (step + 1) == len(dataloader)

        ):

            self.scaler.unscale_(

                self.optimizer

            )

            torch.nn.utils.clip_grad_norm_(

                self.unet.parameters(),

                self.grad_clip,

            )

            self.scaler.step(

                self.optimizer

            )

            self.scaler.update()

            self.optimizer.zero_grad(

                set_to_none=True

            )

            if self.scheduler is not None:

                self.scheduler.step()

            if self.ema is not None:

                self.ema.update(

                    self.unet

                )

        ######################################################
        # Logger
        ######################################################

        self.logger.update(

             **outputs["losses"]

        )

        ######################################################
        # Progress bar
        ######################################################

        pbar.set_postfix(

            loss=f"{outputs['loss'].item():.4f}",

            diff=f"{outputs['losses']['diffusion']:.4f}",

            recon=f"{outputs['losses']['reconstruction']:.4f}",

        )

    ##########################################################
    # Finish epoch
    ##########################################################

    epoch_stats = self.logger.end_epoch()

    return epoch_stats
