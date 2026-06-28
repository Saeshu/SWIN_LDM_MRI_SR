def main():
    import torch
    from torch.cuda.amp import GradScaler
    from torch.utils.data import DataLoader
    from torch.amp import GradScaler
    from Data.dataset import MRIDataset
    from Data.loader import create_dataloaders
    from models.ae import AutoEncoder
    from models.eps_unet3D import ConditionalEpsUNet3D
    
    from Diffusion.LinearNoise import NoiseScheduler
    from Utils.EMA import EMA
    from training.train import DiffusionTrainer
    #from Diffusion.validate import validate
    #from Utils.logger import TrainingLogger

    
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    path = "/workspace/dataset"
    ##########################################################
    # Dataset
    ##########################################################
    
    train_ds = MRIDataset(path)
    
    #val_ds = MRIDataset(...)
    
    train_loader, _ = create_dataloaders(
    data_root="/workspace/dataset",
    batch_size=1,
    crop_size=(32, 128, 128),
    downscale_factor=None,
    num_workers=4,
    val_split=0.0
    )
    
    # val_loader = DataLoader(
    
    #     val_ds,
    
    #     batch_size=1,
    
    #     shuffle=False,
    
    # )
    
    
    
    ##########################################################
    # Diffusion
    ##########################################################
    
    unet = ConditionalEpsUNet3D(
    
        z_ch=256,
    
        cond_ch=256,
    
    ).to(device)
    
    ##########################################################
    # Optimizer
    ##########################################################
    
    optimizer = torch.optim.AdamW(
    
        unet.parameters(),
    
        lr=1e-4,
    
        weight_decay=1e-4,
    
    )
    
    ##########################################################
    # Scheduler
    ##########################################################
    
    noise_scheduler = NoiseScheduler()
    
    ##########################################################
    # EMA
    ##########################################################
    
    ema = EMA(
    
        unet,
    
        decay=0.9999,
    
    )
    
    ##########################################################
    # AMP
    ##########################################################
    
    scaler = GradScaler()

    ##########################################################
    # Autoencoder
    ##########################################################
    
    ae = AutoEncoder().to(device)
    def load_checkpoint(ae, optimizer, scaler, path, device):
        checkpoint = torch.load(path, map_location=device)
    
        ae.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
    
        start_epoch = checkpoint["epoch"] + 1
    
        print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
    
        return start_epoch
    path = '/workspace/ckpt/best8.pt'
    load_checkpoint(ae, optimizer, scaler, path, device)
    # ckpt = torch.load(
    #     "/workspace/ckpt/last8.pt",
    #     map_location=device,
    # )
    
    # ae.load_state_dict(ckpt)
    print("AE parameters:",
          sum(p.numel() for p in ae.parameters()))
    
    print("UNet parameters:",
          sum(p.numel() for p in unet.parameters()))
    ae.eval()
    
    for p in ae.parameters():
    
        p.requires_grad = False
    ##########################################################
    # Validation
    ##########################################################
    
    # validator = DiffusionValidator(
    
    #     ae=ae,
    
    #     ema=ema,
    
    #     scheduler=noise_scheduler,
    
    #     device=device,
    
    # )
    
    ##########################################################
    # Logger
    ##########################################################
    
    #logger = TrainingLogger()
    
    ##########################################################
    # Trainer
    ##########################################################
    
    trainer = DiffusionTrainer(
    
        ae=ae,
    
        unet=unet,
    
        optimizer=optimizer,
    
        noise_scheduler=noise_scheduler,
    
        # validator=validator,
    
        # logger=logger,
    
        ema=ema,
    
        scaler=scaler,
    
        device=device,
    
    )
    
    ##########################################################
    # Callbacks
    ##########################################################
    
    # from Diffusion.trajectory import run_trajectory
    # from Diffusion.moe import analyze_moe
    
    # trainer.callbacks["trajectory"] = run_trajectory
    # trainer.callbacks["moe"] = analyze_moe
    
    ##########################################################
    # Train
    ##########################################################
    
    trainer.fit(
    
        train_loader,
    
        epochs=50
    
    )
if __name__ == "__main__":
    main()
