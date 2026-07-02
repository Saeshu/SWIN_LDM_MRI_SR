from tqdm import tqdm
import torch
import torch.nn.functional as F

def train_epoch(
    self,
    dataloader,
    history,
    stats,
    epoch
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

    
    epoch_stats = {

    "z_hr_std": 0.0,
    "z_lr_std": 0.0,
    "z_res_std": 0.0,
    "noise_std": 0.0,
    "z_noisy_std": 0.0,
    "v_target_std": 0.0,
    "v_pred_std": 0.0,
    "x0_pred_std": 0.0,

}


    num_batch = 0
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
        # print(outputs)
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

        epoch_stats["z_hr_std"] += outputs["z_hr"].std().item()
        epoch_stats["z_lr_std"] += outputs["z_lr"].std().item()
        epoch_stats["z_res_std"] += outputs["z_res"].std().item()
        epoch_stats["noise_std"] += outputs["noise"].std().item()
        epoch_stats["z_noisy_std"] += outputs["z_noisy"].std().item()
        epoch_stats["v_target_std"] += outputs["v_target"].std().item()
        epoch_stats["v_pred_std"] += outputs["v_pred"].std().item()
        epoch_stats["x0_pred_std"] += outputs["x0_pred"].std().item()
        num_batch += 1


        
    
    ##########################################################
    # Finish epoch
    ##########################################################
    n = len(dataloader)
    
    for k in running:
        running[k] /= n
    for key in epoch_stats:
        epoch_stats[key] /= num_batch
        print(key,":", epoch_stats[key])
    stats["z_hr_std"].append(outputs["z_hr"].std().item())

    stats["z_lr_std"].append(outputs["z_lr"].std().item())
    
    stats["z_res_std"].append(outputs["z_res"].std().item())
    
    stats["x0_pred_std"].append(outputs["x0_pred"].std().item())
    
    stats["noise_std"].append(outputs["noise"].std().item())
    
    stats["z_noisy_std"].append(outputs["z_noisy"].std().item())

    stats["v_target_std"].append(outputs["v_target"].std().item())
    
    stats["v_pred_std"].append(outputs["v_pred"].std().item())

    cos = F.cosine_similarity(
    outputs["v_pred"].flatten(1),
    outputs["v_target"].flatten(1),
    dim=1,
    ).mean()

    print("cosine:", cos.item())
    
    stats["v_ratio"].append(
        epoch_stats["v_pred_std"] /
        (epoch_stats["v_target_std"] + 1e-8)
    )
    stats["residual_ratio"].append(
        epoch_stats["z_res_std"] /
        (epoch_stats["noise_std"] + 1e-8)
    )
    history["epoch"].append(epoch + 1)

    history["loss"].append(running["loss"])
    
    history["diffusion"].append(running["diffusion"])
    
    history["reconstruction"].append(running["reconstruction"])
    
    history["residual"].append(running["residual"])
    return running, history, stats
    # epoch_stats = self.logger.end_epoch()

    # return epoch_stats
