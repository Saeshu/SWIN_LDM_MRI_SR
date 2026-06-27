from tqdm import tqdm
import torch


@torch.no_grad()
def validate_epoch(
    self,
    dataloader,
):
    """
    Validate one epoch.

    Parameters
    ----------
    dataloader

    Returns
    -------
    dict
        Validation metrics.
    """

    ##########################################################
    # Evaluation mode
    ##########################################################

    self.unet.eval()

    self.ae.eval()

    if hasattr(self, "adapter"):
        self.adapter.eval()

    ##########################################################
    # Running metrics
    ##########################################################

    running = {}

    ##########################################################
    # Progress bar
    ##########################################################

    pbar = tqdm(

        dataloader,

        leave=False,

        desc="Validation",

    )

    ##########################################################
    # Loop
    ##########################################################

    for batch in pbar:

        ######################################################
        # Batch
        ######################################################

        hr, lr = batch

        hr = hr.to(self.device)

        lr = lr.to(self.device)

        ######################################################
        # Validate
        ######################################################

        metrics = self.validator.validate_step(

            hr,

            lr,

        )

        ######################################################
        # Running averages
        ######################################################

        for key, value in metrics.items():

            if torch.is_tensor(value):

                value = value.item()

            running.setdefault(key, []).append(value)

        ######################################################
        # Progress
        ######################################################

        pbar.set_postfix(

            psnr=f"{metrics['psnr']:.2f}",

            ssim=f"{metrics['ssim']:.4f}",

            l1=f"{metrics['l1']:.4f}",

        )

    ##########################################################
    # Epoch averages
    ##########################################################

    epoch_metrics = {}

    for key, value in running.items():

        epoch_metrics[key] = sum(value) / len(value)

    ##########################################################
    # Logger
    ##########################################################

    self.logger.update_validation(

        epoch_metrics

    )

    return epoch_metrics
