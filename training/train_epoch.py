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
    
    running = {

    "loss": 0.0,

    "diffusion": 0.0,

    "reconstruction": 0.0,

    "residual": 0.0,

    }
    ##########################################################
    # Train mode
    ##########################################################

    self.unet.train()

    if hasattr(self, "adapter"):
        self.adapter.train()

    ##########################################################
    # Reset logger
    ##########################################################

    # self.logger.running.clear()

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

        

        # print(f"\nStep {step}")
        # print("batch type:", type(batch))

        # if isinstance(batch, (list, tuple)):
        #     print("batch length:", len(batch))
        #     for i, x in enumerate(batch):
        #         print(i, type(x))
        #         if torch.is_tensor(x):
        #             print(x.shape)
        # else:
        #     print(batch.shape)

        hr, lr = batch

        # else:

        #     raise ValueError(
        #         "Expected dataloader to return (hr, lr)"
        #     )

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
        # print("loss type:", loss.dtype)
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

            # if self.scheduler is not None:

            #     self.scheduler.step()

            if self.ema is not None:

                self.ema.update(

                    self.unet

                )

        ######################################################
        # Logger
        ######################################################

        # self.logger.update(

        #      **outputs["losses"]

        # )

        ######################################################
        # Progress bar
        ######################################################

        pbar.set_postfix(

          loss=f"{outputs['loss'].item():.4f}",

          mse=f"{outputs['losses']['mse'].item():.4f}",

          recon=f"{outputs['losses']['recon'].item():.4f}",

          res=f"{outputs['losses']['res'].item():.4f}",
          
          perc=f"{outputs['losses']['perc'].item():.4f}"
        )
        running["loss"] += outputs["loss"].item()

        running["diffusion"] += outputs["losses"]["mse"].item()
        
        running["reconstruction"] += outputs["losses"]["recon"].item()
        
        running["residual"] += outputs["losses"]["res"].item()


        history["epoch"].append(epoch + 1)

        history["loss"].append(train_stats["loss"])
        
        history["diffusion"].append(train_stats["diffusion"])
        
        history["reconstruction"].append(train_stats["reconstruction"])
        
        history["residual"].append(train_stats["residual"])
    ##########################################################
    # Finish epoch
    ##########################################################
    n = len(dataloader)

    for k in running:
        running[k] /= n
    
    return running, history
    # epoch_stats = self.logger.end_epoch()

    # return epoch_stats
