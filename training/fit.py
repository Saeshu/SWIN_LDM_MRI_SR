from sampling.visualize_latent_ema import visualize_with_ema
import torch
import pandas as pd
def fit(
    self,
    train_loader,
    # val_loader,
    epochs,
    start_epoch=0,
):
    """
    Main training loop.

    Parameters
    ----------
    train_loader
    val_loader
    epochs
    start_epoch
    """
    
    history = {

    "epoch": [],

    "loss": [],

    "diffusion": [],

    "reconstruction": [],

    "residual": [],

} 
    stats = {

    # --------------------------------------------------
    # Latent statistics
    # --------------------------------------------------

    "z_hr_std": [],

    "z_lr_std": [],

    "z_res_std": [],

    "z_noisy_std": [],
    
    # --------------------------------------------------
    # Noise / prediction statistics
    # --------------------------------------------------

    "noise_std": [],

    "v_target_std": [],

    "v_pred_std": [],

    "x0_pred_std": [],

    # --------------------------------------------------
    # Useful ratios
    # --------------------------------------------------

    "v_ratio": [],          # v_pred_std / v_target_std

    "residual_ratio": [],   # z_res_std / noise_std
    }
    # df = pd.read_csv("baseline_losses.csv")
    
    # history = df.to_dict("list")
    ##########################################################
    # Best validation loss
    ##########################################################

    best_metric = float("inf")

    ##########################################################
    # Epoch loop
    ##########################################################
    sample_hr, sample_lr = next(iter(train_loader))
    for epoch in range(start_epoch, epochs):
        
        print("\n" + "=" * 70)
        print(f"Epoch {epoch+1}/{epochs}")
        print("=" * 70)

        ######################################################
        # Train
        ######################################################
        

        train_stats, train_history, train_stats = self.train_epoch(train_loader, history, stats, epoch)
    
        history = train_history
        stats = train_stats
        if (epoch + 1) % 10 == 0:      # Change to 5 or 10 later

            print("\nGenerating sample...\n")
        
            visualize_with_ema(
        
                ema=self.ema,
        
                ae=self.ae,
        
                noise_sched=self.noise_scheduler,
        
                device=self.device,
        
                lr=sample_lr,
        
                hr=sample_hr,
        
                guidance_scale=1.5,
        
                title=f"Epoch {epoch+1}",
        
            )
        if (epoch + 1) % 25 == 0: 
            torch.save(
            {
                "epoch": epoch,
                "unet": self.unet.state_dict(),
                "ema": self.ema.ema_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            f"/workspace/ckpt/sample_overfit_epoch_{epoch+1}.pt",
            )
        ######################################################
        # Validation
        ######################################################

        # val_stats = self.validate_epoch(val_loader)

        ######################################################
        # Callbacks
        ######################################################

        # use first validation sample
        hr, lr = next(iter(train_loader))

        # self.run_callbacks(
        #     epoch=epoch,
        #     hr=hr,
        #     lr=lr,
        # )

        ######################################################
        # Logger summary
        ######################################################

        print("\nTraining")

        for key, value in train_stats.items():
            print(f"{key:25s}: {value:.5f}")

        
            
        
        # print("\nValidation")

        # for key, value in val_stats.items():
        #     print(f"{key:25s}: {value:.5f}")

        ######################################################
        # Save last checkpoint
        ######################################################

        # save_last(
        #     model=self.unet,
        #     optimizer=self.optimizer,
        #     scheduler=self.scheduler,
        #     scaler=self.scaler,
        #     ema=self.ema,
        #     epoch=epoch,
        #     save_dir=self.save_dir,
        # )

        ######################################################
        # Save best checkpoint
        ######################################################

        # metric = val_stats.get(
        #     "l1",
        #     val_stats.get("loss", 0.0),
        # )

        # best_metric = save_best(
        #     model=self.unet,
        #     metric=metric,
        #     best_metric=best_metric,
        #     optimizer=self.optimizer,
        #     scheduler=self.scheduler,
        #     scaler=self.scaler,
        #     ema=self.ema,
        #     epoch=epoch,
        #     save_dir=self.save_dir,
        # )
        
        
            
    ##########################################################
    # Finished
    ##########################################################

    print("\nTraining complete.")
    df = pd.DataFrame(history)
        
    df.to_csv("overfit_baseline_losses.csv", index=False)
    
    print(df.tail())
    # return self.logger.history
