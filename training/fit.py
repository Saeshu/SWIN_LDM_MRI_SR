from sampling.visualize_latent_ema import visualize_with_ema
import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F

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
        

        losses, train_history, train_stats = self.train_epoch(train_loader, history, stats, epoch)
    
        history = train_history
        stats = train_stats
        if (epoch + 1) % 1 == 0:      # Change to 5 or 10 later

            print("\nGenerating sample...\n")
            pred = sample_ddim(ae = self.ae,
                unet = self.unet,
                noise_scheduler = self.noise_scheduler,
                lr=sample_lr,
                device=self.device)
            visualize_prediction_3d(pred, sample_hr)
            
        if (epoch + 1) % 5 == 0: 
            torch.save(
            {
                "epoch": epoch,
                "unet": self.unet.state_dict(),
                "ema": self.ema.ema_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            f"/workspace/ckpt/sample_x0_epoch_{epoch+1}.pt",
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

        # for key, value in train_stats.items():
        #     print(f"{key:25s}: {value:.5f}")

        
            
        
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





@torch.no_grad()
def sample_ddim(
    ae,
    unet,
    noise_scheduler,
    lr,
    device,
):
    ae.eval()
    unet.eval()

    lr = lr[:1].to(device)

    ####################################################
    # Encode LR
    ####################################################

    z_lr, w_e2 = ae.encode(lr)

    ####################################################
    # IMPORTANT:
    # Upsample exactly as during training
    ####################################################

    z_lr = F.interpolate(
        z_lr,
        size=(
            z_lr.shape[2],
            lr.shape[3] // 2,
            lr.shape[4] // 2,
        ),
        mode="trilinear",
        align_corners=False,
    )

    ####################################################
    # Start from Gaussian noise
    ####################################################

    x = torch.randn_like(z_lr)

    ####################################################
    # Reverse diffusion
    ####################################################

    T = noise_scheduler.num_timesteps

    for timestep in reversed(range(T)):

        t = torch.full(
            (1,),
            timestep,
            device=device,
            dtype=torch.long,
        )

        ###############################################
        # Predict x0
        ###############################################

        x0_pred = unet(
            z=x,
            t=t,
            cond=z_lr,
            w_e2=w_e2,
        )

        ###############################################
        # Last iteration
        ###############################################

        if timestep == 0:
            x = x0_pred
            break

        ###############################################
        # Current alpha
        ###############################################

        alpha_bar = noise_scheduler.alpha_bars[t].view(
            -1,1,1,1,1
        )

        prev_t = torch.clamp(t-1, min=0)

        alpha_bar_prev = noise_scheduler.alpha_bars[
            prev_t
        ].view(-1,1,1,1,1)

        ###############################################
        # Predict epsilon
        ###############################################

        eps_pred = (
            x
            - torch.sqrt(alpha_bar) * x0_pred
        ) / (
            torch.sqrt(1 - alpha_bar) + 1e-8
        )

        ###############################################
        # DDIM update
        ###############################################

        x = (
            torch.sqrt(alpha_bar_prev) * x0_pred
            +
            torch.sqrt(1 - alpha_bar_prev) * eps_pred
        )

    ####################################################
    # Residual -> HR latent
    ####################################################

    z_hr_pred = z_lr + x

    ####################################################
    # Decode
    ####################################################

    pred = ae.decode(
        z_hr_pred,
        w_e2,
    )

    ####################################################
    # Visualization
    ####################################################

    pred = pred.cpu()

    d = pred.shape[2] // 2

    plt.figure(figsize=(5,5))
    plt.imshow(
        pred[0,0,d],
        cmap="gray",
    )
    plt.title("Generated")
    plt.axis("off")
    plt.show()

    ae.train()
    unet.train()

    return pred
@torch.no_grad()
def visualize_prediction_3d(pred, gt):
    
    pred = pred.detach().cpu()
    gt = gt.detach().cpu()
    print("Pred:", pred.shape)
    print("GT  :", gt.shape)
    D_pred, H_pred, W_pred = pred.shape[2:]
    D_gt, H_gt, W_gt = gt.shape[2:]
    
    d_pred, h_pred, w_pred = D_pred//2, H_pred//2, W_pred//2
    d_gt, h_gt, w_gt = D_gt//2, H_gt//2, W_gt//2

    fig, ax = plt.subplots(2, 3, figsize=(15, 10))

    # Ground truth
    ax[0,0].imshow(gt[0,0,d_gt], cmap="gray")
    ax[0,1].imshow(gt[0,0,:,h_gt,:], cmap="gray")
    ax[0,2].imshow(gt[0,0,:,:,w_gt], cmap="gray")
    
    # Prediction
    ax[1,0].imshow(pred[0,0,d_pred], cmap="gray")
    ax[1,1].imshow(pred[0,0,:,h_pred,:], cmap="gray")
    ax[1,2].imshow(pred[0,0,:,:,w_pred], cmap="gray")

    for a in ax.ravel():
        a.axis("off")

    plt.tight_layout()
    plt.show()
