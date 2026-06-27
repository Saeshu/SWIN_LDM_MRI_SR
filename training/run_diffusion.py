import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from Data.dataset import MRIDataset

from models.ae import AutoEncoder
from models.eps_unet3D import ConditionalEpsUNet3D

from Diffusion.LinearNoise import NoiseScheduler
from Diffusion.ema import EMA
from Diffusion.train import DiffusionTrainer
from Diffusion.validate import DiffusionValidator
from Diffusion.logger import TrainingLogger

def main():
    
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    path = "/content/drive/MyDrive/IXI-T1"
    ##########################################################
    # Dataset
    ##########################################################
    
    train_ds = MRIDataset(path)
    
    #val_ds = MRIDataset(...)
    
    train_loader = DataLoader(
    
        train_ds,
    
        batch_size=1,
    
        shuffle=True,
    
        num_workers=4,
    
        pin_memory=True,
    
    )
    
    val_loader = DataLoader(
    
        train_ds,
    
        batch_size=1,
    
        shuffle=False,
    
    )
    
    ##########################################################
    # Autoencoder
    ##########################################################
    
    ae = AutoEncoder().to(device)
    
    ckpt = torch.load(
        "best_autoencoder.pt",
        map_location=device,
    )
    
    ae.load_state_dict(ckpt)
    
    ae.eval()
    
    for p in ae.parameters():
    
        p.requires_grad = False
    
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
    
        beta=0.9999,
    
    )
    
    ##########################################################
    # AMP
    ##########################################################
    
    scaler = GradScaler()
    
    ##########################################################
    # Validation
    ##########################################################
    
    validator = DiffusionValidator(
    
        ae=ae,
    
        ema=ema,
    
        scheduler=noise_scheduler,
    
        device=device,
    
    )
    
    ##########################################################
    # Logger
    ##########################################################
    
    logger = TrainingLogger()
    
    ##########################################################
    # Trainer
    ##########################################################
    
    trainer = DiffusionTrainer(
    
        ae=ae,
    
        unet=unet,
    
        optimizer=optimizer,
    
        noise_scheduler=noise_scheduler,
    
        validator=validator,
    
        logger=logger,
    
        ema=ema,
    
        scaler=scaler,
    
        device=device,
    
    )
    
    ##########################################################
    # Callbacks
    ##########################################################
    
    from Diffusion.trajectory import run_trajectory
    from Diffusion.moe import analyze_moe
    
    trainer.callbacks["trajectory"] = run_trajectory
    trainer.callbacks["moe"] = analyze_moe
    
    ##########################################################
    # Train
    ##########################################################
    
    trainer.fit(
    
        train_loader,
    
        val_loader,
    
        epochs=50,
    
    )
if __name__ == "main":
    main()
