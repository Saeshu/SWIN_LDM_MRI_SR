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

            z_t

            -

            torch.sqrt(
                1 - alpha_bar
            ) * v_pred

        ) / torch.sqrt(
            alpha_bar
        )

    ############################################################
    # Internal functions
    ############################################################

    def _forward(
        self,
        z_hr,
        z_lr,
        z_res,
        w_e2,
    ):
    
        ####################################################
        # timestep
        ####################################################
    
        t = self.sample_timesteps(
            z_res.shape[0]
        )
    
        ####################################################
        # noise
        ####################################################
    
        noise = torch.randn_like(z_res)
    
        z_noisy = self.noise_scheduler.add_noise(
            z_res,
            t,
            noise,
        )
    
        ####################################################
        # target
        ####################################################
    
        alpha_bar = self.noise_scheduler.alpha_bars[t].view(
            -1, 1, 1, 1, 1
        )
    
        v_target = (
            torch.sqrt(alpha_bar) * noise
            - torch.sqrt(1 - alpha_bar) * z_res
        )
    
        ####################################################
        # UNet
        ####################################################
    
        with torch.cuda.amp.autocast():
    
            v_pred = self.unet(
                z=z_noisy,
                t=t,
                cond=z_lr,
                w_e2=w_e2,
                alpha=self.cfg_scale,
            )
    
            x0_pred = self.predict_x0(
                z_noisy,
                v_pred,
                alpha_bar,
            )
    
        return {
    
            "t": t,
    
            "noise": noise,
    
            "alpha_bar": alpha_bar,
    
            "z_noisy": z_noisy,
    
            "v_target": v_target,
    
            "v_pred": v_pred,
    
            "x0_pred": x0_pred,
    
        }
    
        
    
        ############################################################
    
    def _compute_loss(
            self,
            outputs,
        ):
    
            return compute_losses(
    
                v_pred=outputs["v_pred"],
    
                v_target=outputs["v_target"],
    
                x0_pred=outputs["x0_pred"],
    
                residual_gt=outputs["z_res"],
    
            )

    def _optimizer_step(
        self,
        loss,
    ):
    
        self.optimizer.zero_grad(
            set_to_none=True
        )
    
        self.scaler.scale(loss).backward()
    
        self.scaler.unscale_(self.optimizer)
    
        torch.nn.utils.clip_grad_norm_(
    
            self.unet.parameters(),
    
            self.grad_clip,
    
        )
    
        self.scaler.step(self.optimizer)
    
        self.scaler.update()
    
        if self.ema is not None:
    
            self.ema.update(self.unet) 
    ############################################################
    # Core training
    ############################################################

import torch
import torch.nn.functional as F


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
    
        self._optimizer_step(
    
            losses["total"]
    
        )
    
        return losses

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
