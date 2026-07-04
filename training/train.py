"""
============================================================
train.py

Diffusion Trainer

Author:
    Diffusion Framework
============================================================
"""

import torch
from torch.cuda.amp import GradScaler

from Diffusion.advanced_losses import compute_losses
from training.train_epoch import train_epoch
from training.train_step import train_step
from training.fit import fit
import torch
import torch.nn.functional as F
# from Utils.checkpoint import save_best, save_last


class DiffusionTrainer:

    ############################################################
    # Initialization
    ############################################################

    def __init__(
        self,
        ae,
        unet,
        optimizer,
        noise_scheduler,
        # validator,
        # logger,
        device,
        ema=None,
        scheduler=None,
        scaler=None,
        save_dir="./checkpoints",
        accum_steps=1,
        grad_clip=1.0,
        cfg_scale=1.0,
    ):

        self.ae = ae
        self.unet = unet

        self.optimizer = optimizer
        # self.scheduler = scheduler
        self.noise_scheduler = noise_scheduler
        
        # self.validator = validator
        # self.logger = logger

        self.device = device

        self.ema = ema

        self.scaler = (
            scaler
            if scaler is not None
            else GradScaler()
        )

        self.save_dir = save_dir

        self.accum_steps = accum_steps
        self.grad_clip = grad_clip

        self.cfg_scale = cfg_scale
            
        ########################################################

        # self.callbacks = {

        #     "trajectory": None,

        #     "moe": None,

        #     "representation": None,

        #     "sample": None,

        # }

        ########################################################

        # self.callback_frequency = {

        #     "trajectory": 5,

        #     "moe": 5,

        #     "representation": 10,

        #     "sample": 10,

        # }

    ############################################################
    # Utilities
    ############################################################

    def sample_timesteps(
        self,
        batch_size,
    ):

        T = self.noise_scheduler.num_timesteps

        u = torch.rand(
            batch_size,
            device=self.device,
        )

        t = (
            (u ** 2) * T
        ).long()

        return torch.clamp(
            t,
            max=T - 1,
        )

    ############################################################

    def predict_x0(
        self,
        z_t,
        v_pred,
        alpha_bar,
    ):
    
        return (
    
            torch.sqrt(alpha_bar) * z_t
    
            -
    
            torch.sqrt(1.0 - alpha_bar) * v_pred
    
        )
    ############################################################
    # Internal functions
    ############################################################
    import torch.nn.functional as F

    def _encode(
        self,
        hr,
        lr,
    ):
        """
        Encode HR and LR images into latent space.

        Returns
        -------
        dict containing:
            z_hr
            z_lr
            z_res
            w_e2
        """

        with torch.no_grad():

            ####################################################
            # Encode
            ####################################################

            z_hr, w_e2 = self.ae.encode(hr)

            z_lr, _ = self.ae.encode(lr)

            ####################################################
            # Match latent resolution
            ####################################################

            if z_lr.shape[2:] != z_hr.shape[2:]:

                z_lr = F.interpolate(

                    z_lr,

                    size=z_hr.shape[2:],

                    mode="trilinear",

                    align_corners=False,

                )

            ####################################################
            # Residual latent
            ####################################################

            z_res = z_hr - z_lr

        return {

            "z_hr": z_hr,

            "z_lr": z_lr,

            "z_res": z_res,

            "w_e2": w_e2,

        }
    
    def _forward(
        self,
        encoded
    ):

        z_hr = encoded["z_hr"]
        z_lr = encoded["z_lr"]
        z_res = encoded["z_res"]
        w_e2 = encoded["w_e2"]
        ####################################################
        # timestep
        ####################################################
    
        # t = self.sample_timesteps(
        #     batch_size=z_hr.shape[0],
        #     epoch=self.current_epoch,
        #     total_epochs=self.total_epochs,
        # )
        t = torch.full(
            (z_hr.shape[0],),
            0,
            device=z_hr.device,
            dtype=torch.long,
                )
        ####################################################
        # noise
        ####################################################
        
        noise = torch.randn_like(z_res)
        # print("t:", t)
        # print("max t:", t.max().item())
        # print("num_timesteps:", self.noise_scheduler.num_timesteps)
        z_noisy = self.noise_scheduler.add_noise(
            z_res,
            t,
            noise,
        )
        
        ####################################################
        # target
        ####################################################
        alpha_bar = self.noise_scheduler.alpha_bars[t]
        # print("t.device:", t.device)

        #     print("alpha_bars.device:", self.noise_scheduler.alpha_bars.device)
            
        #     print("betas.device:", self.noise_scheduler.betas.device)
            
        #     print("alphas.device:", self.noise_scheduler.alphas.device)
        alpha_bar = self.noise_scheduler.alpha_bars[t].view(
            -1, 1, 1, 1, 1
        )
        
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus = torch.sqrt(1.0 - alpha_bar)
        v_target = (
            torch.sqrt(alpha_bar) * noise
            - torch.sqrt(1 - alpha_bar) * z_res
        )
    
        ####################################################
        # UNet
        ####################################################
    
        with torch.cuda.amp.autocast():
    
            x0_pred = self.unet(
                z=z_noisy,
                t=t,
                cond=z_lr,
                w_e2=w_e2,
                alpha=self.cfg_scale,
            )

            v_from_x0 = (
                sqrt_alpha_bar * noise
                - sqrt_one_minus * x0_pred
            )

            
    
        return {

          **encoded,

          "v_pred": v_from_x0,

          "v_target": v_target,

          "x0_pred": x0_pred,

          # Useful for diagnostics / future losses
          "t": t,

          "noise": noise,

          "alpha_bar": alpha_bar,

          "z_noisy": z_noisy,

      }
    
        
    
        ############################################################
    
    def sample_timesteps(self, batch_size, epoch=None, total_epochs=None):

        if epoch is None or total_epochs is None:
            return torch.randint(
                0,
                self.noise_scheduler.num_timesteps,
                (batch_size,),
                device=self.device,
            )
    
        progress = epoch / total_epochs
    
        min_t = int(
            (1.0 - progress)
            * (self.noise_scheduler.num_timesteps - 1)
        )
    
        curriculum_prob = 0.8
        
        if torch.rand(1).item() < curriculum_prob:
    
            return torch.randint(
                min_t,
                self.noise_scheduler.num_timesteps,
                (batch_size,),
                device=self.device,
            )
    
        else:
    
            return torch.randint(
                0,
                self.noise_scheduler.num_timesteps,
                (batch_size,),
                device=self.device,
            )

       
    def _compute_loss(
        ae,
        outputs,
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
       
        x0_pred=outputs["x0_pred"]

        z_res=outputs["z_res"]

        z_hr=outputs["z_hr"]

        z_lr=outputs["z_lr"]

        v_pred=outputs["v_pred"]

        v_target=outputs["v_target"]
        
        with torch.cuda.amp.autocast():
            mse_loss = F.mse_loss(
                x0_pred,
                z_res,
            )
    
            ####################################################
            # Residual loss
            ####################################################
            
            
            v_loss = F.l1_loss(
                v_pred,
                v_target,
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
    
                0.1 * v_loss
    
                +
    
                0.01 * recon_loss
    
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

            "res": v_loss,

            "perc": perc_loss,

        }
    

    # def _optimizer_step(
    #     self,
    #     loss,
    # ):
    
    #     self.optimizer.zero_grad(
    #         set_to_none=True
    #     )
    
    #     self.scaler.scale(loss).backward()
    
    #     self.scaler.unscale_(self.optimizer)
    
    #     torch.nn.utils.clip_grad_norm_(
    
    #         self.unet.parameters(),
    
    #         self.grad_clip,
    
    #     )
    
    #     self.scaler.step(self.optimizer)
    
    #     self.scaler.update()
    
    #     if self.ema is not None:
    
    #         self.ema.update(self.unet) 
    ############################################################
    # Core training
    ############################################################




    def train_step(
        self,
        hr,
        lr,
    ):
    
        hr = hr.to(self.device)
    
        lr = lr.to(self.device)
    
        ####################################################
        # Encode
        ####################################################
    
        with torch.no_grad():
    
            z_hr, w_e2 = self.ae.encode(hr)
    
            z_lr, _ = self.ae.encode(lr)
    
            z_lr = F.interpolate(
    
                z_lr,
    
                size=z_hr.shape[2:],
    
                mode="trilinear",
    
                align_corners=False,
    
            )
    
        z_res = z_hr - z_lr
    
        ####################################################
        # Forward
        ####################################################
    
        outputs = self._forward(
    
            z_hr,
    
            z_lr,
    
            z_res,
    
            w_e2,
    
        )
    
        ####################################################
        # Loss
        ####################################################
    
        losses = self._compute_loss({
    
            **outputs,
    
            "z_res": z_res,
    
        })
    
        ####################################################
        # Optimizer
        ####################################################
    
        # self._optimizer_step(
    
        #     losses["total"]
    
        # )
    
        # return losses

    ############################################################

    def train_epoch(
        self,
        dataloader,
    ):
        raise NotImplementedError

    ############################################################

    # @torch.no_grad()
    # def validate_epoch(
    #     self,
    #     dataloader,
    # ):
    #     raise NotImplementedError

    ############################################################

    # @torch.no_grad()
    # def run_callbacks(
    #     self,
    #     epoch,
    #     hr,
    #     lr,
    # ):
    #     raise NotImplementedError

    ############################################################
    # Training Loop
    ############################################################

    def fit(
        self,
        train_loader,
        # val_loader,
        epochs,
        start_epoch=0,
    ):

        ########################################################
        # Fixed callback batch
        ########################################################

        # callback_hr, callback_lr = next(
        #     iter(val_loader)
        # )

        # best_metric = float("inf")

        ########################################################

        for epoch in range(
            start_epoch,
            epochs,
        ):

            print(
                "\n"
                + "=" * 70
            )

            print(
                f"Epoch {epoch+1}/{epochs}"
            )

            print(
                "=" * 70
            )

            ####################################################
            # Train
            ####################################################

            train_stats = self.train_epoch(
                train_loader
            )

            ####################################################
            # Validation
            ####################################################

            # val_stats = self.validate_epoch(
            #     val_loader
            # )

            ####################################################
            # Callbacks
            ####################################################

            # self.run_callbacks(

            #     epoch,

            #     callback_hr,

            #     callback_lr,

            # )

            ####################################################
            # Console summary
            ####################################################

            print("\nTraining")

            for k, v in train_stats.items():

                print(
                    f"{k:25s}: {v:.5f}"
                )

            # print("\nValidation")

            # for k, v in val_stats.items():

            #     print(
            #         f"{k:25s}: {v:.5f}"
            #     )

            ####################################################
            # Save checkpoints
            ####################################################

        #     save_last(

        #         model=self.unet,

        #         optimizer=self.optimizer,

        #         scheduler=self.scheduler,

        #         scaler=self.scaler,

        #         ema=self.ema,

        #         epoch=epoch,

        #         save_dir=self.save_dir,

        #     )

        #     metric = val_stats.get(

        #         "l1",

        #         val_stats.get(

        #             "loss",

        #             0.0,

        #         ),

        #     )

        #     best_metric = save_best(

        #         model=self.unet,

        #         metric=metric,

        #         best_metric=best_metric,

        #         optimizer=self.optimizer,

        #         scheduler=self.scheduler,

        #         scaler=self.scaler,

        #         ema=self.ema,

        #         epoch=epoch,

        #         save_dir=self.save_dir,

        #     )

        #     ####################################################
        #     # Epoch summary
        #     ####################################################

        #     # self.logger.summary()

        # ########################################################

        print(
            "\nTraining complete."
        )

        # return self.logger.history
DiffusionTrainer.train_step = train_step
DiffusionTrainer.train_epoch = train_epoch
DiffusionTrainer.fit = fit
