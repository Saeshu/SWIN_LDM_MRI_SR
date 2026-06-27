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

from Diffusion.losses import compute_losses
from Diffusion.checkpoint import save_best, save_last


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
        validator,
        logger,
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
        self.scheduler = scheduler
        self.noise_scheduler = noise_scheduler

        self.validator = validator
        self.logger = logger

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

        self.callbacks = {

            "trajectory": None,

            "moe": None,

            "representation": None,

            "sample": None,

        }

        ########################################################

        self.callback_frequency = {

            "trajectory": 5,

            "moe": 5,

            "representation": 10,

            "sample": 10,

        }

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

    def _encode(
        self,
        hr,
        lr,
    ):
        raise NotImplementedError

    ############################################################

    def _forward(
        self,
        encoded,
    ):
        raise NotImplementedError

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

    ############################################################
    # Core training
    ############################################################

    def train_step(
        self,
        hr,
        lr,
        debug=False,
    ):
        raise NotImplementedError

    ############################################################

    def train_epoch(
        self,
        dataloader,
    ):
        raise NotImplementedError

    ############################################################

    @torch.no_grad()
    def validate_epoch(
        self,
        dataloader,
    ):
        raise NotImplementedError

    ############################################################

    @torch.no_grad()
    def run_callbacks(
        self,
        epoch,
        hr,
        lr,
    ):
        raise NotImplementedError

    ############################################################
    # Training Loop
    ############################################################

    def fit(
        self,
        train_loader,
        val_loader,
        epochs,
        start_epoch=0,
    ):

        ########################################################
        # Fixed callback batch
        ########################################################

        callback_hr, callback_lr = next(
            iter(val_loader)
        )

        best_metric = float("inf")

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

            val_stats = self.validate_epoch(
                val_loader
            )

            ####################################################
            # Callbacks
            ####################################################

            self.run_callbacks(

                epoch,

                callback_hr,

                callback_lr,

            )

            ####################################################
            # Console summary
            ####################################################

            print("\nTraining")

            for k, v in train_stats.items():

                print(
                    f"{k:25s}: {v:.5f}"
                )

            print("\nValidation")

            for k, v in val_stats.items():

                print(
                    f"{k:25s}: {v:.5f}"
                )

            ####################################################
            # Save checkpoints
            ####################################################

            save_last(

                model=self.unet,

                optimizer=self.optimizer,

                scheduler=self.scheduler,

                scaler=self.scaler,

                ema=self.ema,

                epoch=epoch,

                save_dir=self.save_dir,

            )

            metric = val_stats.get(

                "l1",

                val_stats.get(

                    "loss",

                    0.0,

                ),

            )

            best_metric = save_best(

                model=self.unet,

                metric=metric,

                best_metric=best_metric,

                optimizer=self.optimizer,

                scheduler=self.scheduler,

                scaler=self.scaler,

                ema=self.ema,

                epoch=epoch,

                save_dir=self.save_dir,

            )

            ####################################################
            # Epoch summary
            ####################################################

            self.logger.summary()

        ########################################################

        print(
            "\nTraining complete."
        )

        return self.logger.history
