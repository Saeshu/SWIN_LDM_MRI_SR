import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Updated imports for our shaped architecture
from models.ShapedEncoder3D import ShapedEncoder3D
from models.ShapedDecoder3D import SimpleShapedDecoder3D, ShapedDecoder3D
from Data.dataset import MRIDataset
from models.utils import save_checkpoint


def train_autoencoder(
    data_root,
    device="cuda",
    batch_size=1,
    lr=2e-4,
    num_steps=20000,
    log_every=100,
    save_every=2000,
    use_skip_connections=True,  # NEW: Toggle skip connections
    base_ch=16,                  # NEW: Control model capacity
    checkpoint_path=None,        # NEW: Resume training
):
    """
    Train the shaped autoencoder for MRI reconstruction.
    
    Args:
        data_root: Path to training data
        device: 'cuda' or 'cpu'
        batch_size: Batch size (keep at 1 for 3D volumes due to memory)
        lr: Learning rate
        num_steps: Total training steps
        log_every: Log frequency
        save_every: Checkpoint save frequency
        use_skip_connections: If True, uses ShapedDecoder3D (with skips),
                              If False, uses SimpleShapedDecoder3D (no skips)
        base_ch: Base channel count (16 or 32 for more capacity)
        checkpoint_path: Path to resume from checkpoint
    """
    
    os.makedirs("checkpoints", exist_ok=True)
    
    # -------------------------
    # Dataset
    # -------------------------
    dataset = MRIDataset(data_root)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    
    # -------------------------
    # Models - CHANGED: Use shaped architecture
    # -------------------------
    # Encoder: Input (1 ch) → 16 ch → 32 ch
    encoder = ShapedEncoder3D(in_ch=1, base_ch=base_ch).to(device)
    
    # Decoder: Choose based on skip connection preference
    if use_skip_connections:
        # Production decoder with U-Net skip connections
        decoder = ShapedDecoder3D(out_ch=1, base_ch=base_ch).to(device)
        print(f"[INFO] Using ShapedDecoder3D with skip connections")
    else:
        # Simple decoder for initial testing
        decoder = SimpleShapedDecoder3D(out_ch=1, base_ch=base_ch).to(device)
        print(f"[INFO] Using SimpleShapedDecoder3D (no skip connections)")
    
    encoder.train()
    decoder.train()
    
    # -------------------------
    # Optimizer
    # -------------------------
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    
    # NEW: Learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_steps,
        eta_min=lr * 0.01
    )
    
    # -------------------------
    # Resume from checkpoint (NEW)
    # -------------------------
    start_step = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[INFO] Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_step = checkpoint['step']
        print(f"[INFO] Resumed from step {start_step}")
    
    # -------------------------
    # Model info (NEW)
    # -------------------------
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "="*70)
    print("Model Statistics")
    print("="*70)
    print(f"Encoder parameters:  {count_parameters(encoder):,}")
    print(f"Decoder parameters:  {count_parameters(decoder):,}")
    print(f"Total parameters:    {count_parameters(encoder) + count_parameters(decoder):,}")
    print(f"Base channels:       {base_ch}")
    print(f"Skip connections:    {use_skip_connections}")
    print("="*70 + "\n")
    
    # -------------------------
    # Training loop
    # -------------------------
    step = start_step
    pbar = tqdm(initial=start_step, total=num_steps, desc="Training")
    
    # NEW: Track metrics for monitoring
    running_loss = 0.0
    best_loss = float('inf')
    
    while step < num_steps:
        for hr in loader:
            hr = hr.to(device)
            
            # CHANGED: Encoder now returns (latent, skip_connections)
            latent, skip1 = encoder(hr)
            
            # CHANGED: Decoder uses different forward signature
            if use_skip_connections:
                # Pass skip connections to decoder
                recon = decoder(latent, skip1)
            else:
                # Simple decoder doesn't use skips
                recon = decoder(latent)
            
            # Compute loss (L1 is good for MRI - preserves edges)
            loss = F.l1_loss(recon, hr)
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            
            # NEW: Gradient clipping for stability with SWIN attention
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            
            optimizer.step()
            scheduler.step()  # NEW: Update learning rate
            
            # Track metrics
            running_loss += loss.item()
            
            # -------------------------
            # Logging (IMPROVED)
            # -------------------------
            if step % log_every == 0:
                avg_loss = running_loss / log_every if step > 0 else loss.item()
                current_lr = scheduler.get_last_lr()[0]
                
                print(f"[AE] step {step:06d} | loss {avg_loss:.4e} | lr {current_lr:.2e}")
                pbar.set_postfix({'loss': f'{avg_loss:.4e}', 'lr': f'{current_lr:.2e}'})
                
                running_loss = 0.0
            
            # -------------------------
            # Checkpointing (IMPROVED)
            # -------------------------
            if step % save_every == 0 and step > 0:
                checkpoint_data = {
                    'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict(),
                    'optimizer': optimizer.state_dict(),  # NEW: Save optimizer state
                    'scheduler': scheduler.state_dict(),  # NEW: Save scheduler state
                    'step': step,
                    'loss': loss.item(),
                    'config': {  # NEW: Save configuration
                        'base_ch': base_ch,
                        'use_skip_connections': use_skip_connections,
                        'lr': lr,
                    }
                }
                
                # Save regular checkpoint
                checkpoint_name = f"checkpoints/ae_step_{step:06d}.pt"
                save_checkpoint(checkpoint_data, checkpoint_name)
                print(f"[INFO] Saved checkpoint: {checkpoint_name}")
                
                # NEW: Save best model
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_checkpoint_name = "checkpoints/ae_best.pt"
                    save_checkpoint(checkpoint_data, best_checkpoint_name)
                    print(f"[INFO] New best loss: {best_loss:.4e}")
            
            step += 1
            pbar.update(1)
            
            if step >= num_steps:
                break
    
    pbar.close()
    
    # -------------------------
    # Final save
    # -------------------------
    final_checkpoint = {
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'step': step,
        'config': {
            'base_ch': base_ch,
            'use_skip_connections': use_skip_connections,
            'lr': lr,
        }
    }
    save_checkpoint(final_checkpoint, "checkpoints/ae_final.pt")
    print(f"\n[INFO] Training complete! Final checkpoint saved.")


def validate_autoencoder(
    checkpoint_path,
    data_root,
    device="cuda",
    num_samples=10,
    use_skip_connections=True,
    base_ch=16,
):
    """
    NEW: Validation function to test reconstruction quality.
    
    Args:
        checkpoint_path: Path to trained checkpoint
        data_root: Path to validation data
        device: 'cuda' or 'cpu'
        num_samples: Number of samples to validate on
        use_skip_connections: Must match training configuration
        base_ch: Must match training configuration
    """
    print("\n" + "="*70)
    print("Validation")
    print("="*70)
    
    # Load dataset
    dataset = MRIDataset(data_root)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Load models
    encoder = ShapedEncoder3D(in_ch=1, base_ch=base_ch).to(device)
    if use_skip_connections:
        decoder = ShapedDecoder3D(out_ch=1, base_ch=base_ch).to(device)
    else:
        decoder = SimpleShapedDecoder3D(out_ch=1, base_ch=base_ch).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    
    encoder.eval()
    decoder.eval()
    
    # Validate
    total_loss = 0.0
    with torch.no_grad():
        for i, hr in enumerate(loader):
            if i >= num_samples:
                break
            
            hr = hr.to(device)
            latent, skip1 = encoder(hr)
            
            if use_skip_connections:
                recon = decoder(latent, skip1)
            else:
                recon = decoder(latent)
            
            loss = F.l1_loss(recon, hr)
            total_loss += loss.item()
            
            print(f"Sample {i+1}/{num_samples} | Loss: {loss.item():.4e}")
    
    avg_loss = total_loss / num_samples
    print(f"\nAverage validation loss: {avg_loss:.4e}")
    print("="*70)
    
    return avg_loss


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    
    # -------------------------
    # Configuration
    # -------------------------
    config = {
        "data_root": "/content/drive/MyDrive/mri_dataset/train",
        "device": device,
        "batch_size": 1,           # Keep at 1 for 3D volumes (memory constraint)
        "lr": 2e-4,                # Good starting point
        "num_steps": 20000,
        "log_every": 100,
        "save_every": 2000,
        "use_skip_connections": True,  # Set False to test SimpleShapedDecoder3D first
        "base_ch": 16,             # Use 32 for more capacity (but 2x memory)
        "checkpoint_path": None,   # Set to resume training
    }
    
    # -------------------------
    # Training
    # -------------------------
    train_autoencoder(**config)
    
    # -------------------------
    # Optional: Validation (uncomment to run)
    # -------------------------
    # validate_autoencoder(
    #     checkpoint_path="checkpoints/ae_best.pt",
    #     data_root="/content/drive/MyDrive/mri_dataset/val",
    #     device=device,
    #     num_samples=10,
    #     use_skip_connections=config["use_skip_connections"],
    #     base_ch=config["base_ch"],
    # )
