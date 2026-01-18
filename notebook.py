# ============================================================================
# Cell 1: Setup and Installations
# ============================================================================

# Install required packages (run once)
!pip install torch torchvision einops -q

# If using Google Colab, mount drive
from google.colab import drive
drive.mount('/content/drive')

print("✓ Setup complete!")


# ============================================================================
# Cell 2: Project Structure Setup
# ============================================================================

import os
import sys

# Define your project root (adjust this path!)
PROJECT_ROOT = "/content/drive/MyDrive/mri_super_resolution"

# Create directory structure if it doesn't exist
os.makedirs(f"{PROJECT_ROOT}/models", exist_ok=True)
os.makedirs(f"{PROJECT_ROOT}/Data", exist_ok=True)
os.makedirs(f"{PROJECT_ROOT}/checkpoints", exist_ok=True)
os.makedirs(f"{PROJECT_ROOT}/outputs", exist_ok=True)

# Add project root to Python path so we can import modules
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print(f"Project root: {PROJECT_ROOT}")
print(f"Python path updated: {PROJECT_ROOT in sys.path}")

# Verify directory structure
print("\nDirectory structure:")
for folder in ['models', 'Data', 'checkpoints', 'outputs']:
    path = f"{PROJECT_ROOT}/{folder}"
    exists = "✓" if os.path.exists(path) else "✗"
    print(f"  {exists} {folder}/")


# ============================================================================
# Cell 3: Save Your Model Files
# ============================================================================

# IMPORTANT: Save your model code as .py files in the models/ directory
# You can do this in two ways:

# Method 1: Use %%writefile magic command (see next cells)
# Method 2: Upload .py files directly to Google Drive

print("Ready to save model files!")
print(f"Save location: {PROJECT_ROOT}/models/")


# ============================================================================
# Cell 4: Save ShapedEncoder3D.py
# ============================================================================

%%writefile {PROJECT_ROOT}/models/ShapedEncoder3D.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# [PASTE YOUR FULL SWIN_3D_MRI CODE HERE - from the artifact]
# Include: window_partition_3d, window_reverse_3d, WindowAttention3D, 
# SwinTransformerBlock3D, KernelBasis3D, SwinMixingField3D, 
# AttentionShapedConv3D, SwinGuidedConvBlock3D, ShapedDownBlock3D, ShapedEncoder3D

# NOTE: You'll copy-paste the entire artifact code here
# For brevity in this example, I'm showing the structure


# ============================================================================
# Cell 5: Save ShapedDecoder3D.py
# ============================================================================

%%writefile {PROJECT_ROOT}/models/ShapedDecoder3D.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the encoder components we need
from models.ShapedEncoder3D import SwinGuidedConvBlock3D

def match_shape(x, ref):
    """Match x spatial shape to ref by symmetric padding or cropping."""
    _, _, D, H, W = ref.shape
    d, h, w = x.shape[2:]
    pd, ph, pw = D - d, H - h, W - w
    
    if pd > 0 or ph > 0 or pw > 0:
        x = F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2, pd // 2, pd - pd // 2))
    
    return x[:, :, :D, :H, :W]


class ShapedUpBlock3D(nn.Module):
    """Upsampling block using SWIN-guided shaped convolutions."""
    def __init__(self, in_ch, out_ch, window_size=4):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.shaped = SwinGuidedConvBlock3D(out_ch, out_ch, window_size=window_size)
        
    def forward(self, x):
        x = self.upsample(x)
        x = self.shaped(x)
        return x


class SimpleShapedDecoder3D(nn.Module):
    """Simple decoder without skip connections for testing."""
    def __init__(self, out_ch=1, base_ch=16):
        super().__init__()
        self.up1 = ShapedUpBlock3D(base_ch * 2, base_ch)
        self.up2 = ShapedUpBlock3D(base_ch, base_ch)
        self.out = nn.Conv3d(base_ch, out_ch, kernel_size=1)
        
    def forward(self, z):
        x = self.up1(z)
        x = self.up2(x)
        return self.out(x)


class ShapedDecoder3D(nn.Module):
    """Decoder with skip connections for U-Net architecture."""
    def __init__(self, out_ch=1, base_ch=16, window_size=4):
        super().__init__()
        
        self.up1 = nn.ConvTranspose3d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.fuse1 = SwinGuidedConvBlock3D(base_ch * 2, base_ch, window_size=window_size)
        
        self.up2 = nn.ConvTranspose3d(base_ch, base_ch, kernel_size=2, stride=2)
        self.final_conv = SwinGuidedConvBlock3D(base_ch, base_ch, window_size=window_size)
        
        self.out = nn.Conv3d(base_ch, out_ch, kernel_size=1)
        
    def forward(self, z, skip1=None):
        x = self.up1(z)
        
        if skip1 is None:
            skip1 = torch.zeros_like(x)
        
        skip1 = match_shape(skip1, x)
        x = torch.cat([x, skip1], dim=1)
        x = self.fuse1(x)
        
        x = self.up2(x)
        x = self.final_conv(x)
        return self.out(x)


# ============================================================================
# Cell 6: Save utils.py
# ============================================================================

%%writefile {PROJECT_ROOT}/models/utils.py

import torch

def save_checkpoint(state, filename):
    """Save model checkpoint."""
    torch.save(state, filename)
    print(f"Checkpoint saved: {filename}")

def load_checkpoint(filename, encoder, decoder, optimizer=None, device='cuda'):
    """Load model checkpoint."""
    checkpoint = torch.load(filename, map_location=device)
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    step = checkpoint.get('step', 0)
    print(f"Checkpoint loaded from step {step}")
    return step


# ============================================================================
# Cell 7: Create Dummy Dataset (or use your real dataset)
# ============================================================================

%%writefile {PROJECT_ROOT}/Data/dataset.py

import torch
from torch.utils.data import Dataset
import os
import nibabel as nib
import numpy as np

class MRIDataset(Dataset):
    """
    MRI Dataset loader.
    Assumes .nii or .nii.gz files in the data_root directory.
    """
    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform
        
        # Find all .nii and .nii.gz files
        self.files = []
        for fname in os.listdir(data_root):
            if fname.endswith('.nii') or fname.endswith('.nii.gz'):
                self.files.append(os.path.join(data_root, fname))
        
        print(f"Found {len(self.files)} MRI volumes in {data_root}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Load MRI volume
        fpath = self.files[idx]
        img = nib.load(fpath)
        data = img.get_fdata()
        
        # Normalize to [0, 1]
        data = (data - data.min()) / (data.max() - data.min() + 1e-8)
        
        # Add channel dimension: (D, H, W) -> (1, D, H, W)
        data = torch.from_numpy(data).float().unsqueeze(0)
        
        if self.transform:
            data = self.transform(data)
        
        return data


# Alternative: Dummy dataset for testing without real data
class DummyMRIDataset(Dataset):
    """Generates random MRI-like volumes for testing."""
    def __init__(self, num_samples=100, volume_size=(64, 64, 64)):
        self.num_samples = num_samples
        self.volume_size = volume_size
        print(f"Created dummy dataset with {num_samples} samples of size {volume_size}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random volume
        data = torch.randn(1, *self.volume_size)
        # Apply sigmoid to get values in [0, 1]
        data = torch.sigmoid(data)
        return data


# ============================================================================
# Cell 8: Import Everything and Test
# ============================================================================

import torch
import torch.nn.functional as F

# Import your models
from models.ShapedEncoder3D import ShapedEncoder3D
from models.ShapedDecoder3D import SimpleShapedDecoder3D, ShapedDecoder3D
from models.utils import save_checkpoint, load_checkpoint

# Import dataset
from Data.dataset import MRIDataset, DummyMRIDataset

print("✓ All imports successful!")

# Quick test
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Create models
encoder = ShapedEncoder3D(in_ch=1, base_ch=16).to(device)
decoder = ShapedDecoder3D(out_ch=1, base_ch=16).to(device)

# Test forward pass
test_input = torch.randn(1, 1, 64, 64, 64).to(device)
latent, skip1 = encoder(test_input)
output = decoder(latent, skip1)

print(f"\nTest forward pass:")
print(f"  Input shape:  {test_input.shape}")
print(f"  Latent shape: {latent.shape}")
print(f"  Skip shape:   {skip1.shape}")
print(f"  Output shape: {output.shape}")
print(f"  ✓ Shapes match: {output.shape == test_input.shape}")


# ============================================================================
# Cell 9: Training Setup
# ============================================================================

from torch.utils.data import DataLoader
from tqdm.notebook import tqdm  # Use notebook version for Jupyter

def train_autoencoder(
    data_root,
    device="cuda",
    batch_size=1,
    lr=2e-4,
    num_steps=1000,  # Reduced for notebook testing
    log_every=50,
    save_every=200,
    use_skip_connections=True,
    base_ch=16,
):
    """Train the shaped autoencoder."""
    
    os.makedirs(f"{PROJECT_ROOT}/checkpoints", exist_ok=True)
    
    # Dataset - use dummy for testing, replace with MRIDataset for real data
    dataset = DummyMRIDataset(num_samples=50, volume_size=(64, 64, 64))
    # dataset = MRIDataset(data_root)  # Uncomment for real data
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Models
    encoder = ShapedEncoder3D(in_ch=1, base_ch=base_ch).to(device)
    if use_skip_connections:
        decoder = ShapedDecoder3D(out_ch=1, base_ch=base_ch).to(device)
    else:
        decoder = SimpleShapedDecoder3D(out_ch=1, base_ch=base_ch).to(device)
    
    encoder.train()
    decoder.train()
    
    # Optimizer
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    
    # Training loop
    step = 0
    pbar = tqdm(total=num_steps, desc="Training")
    losses = []
    
    while step < num_steps:
        for hr in loader:
            hr = hr.to(device)
            
            # Forward pass
            latent, skip1 = encoder(hr)
            if use_skip_connections:
                recon = decoder(latent, skip1)
            else:
                recon = decoder(latent)
            
            # Loss
            loss = F.l1_loss(recon, hr)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            
            losses.append(loss.item())
            
            # Logging
            if step % log_every == 0:
                avg_loss = sum(losses[-log_every:]) / min(len(losses), log_every)
                pbar.set_postfix({'loss': f'{avg_loss:.4e}'})
            
            # Save checkpoint
            if step % save_every == 0 and step > 0:
                save_checkpoint({
                    'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': step,
                }, f"{PROJECT_ROOT}/checkpoints/ae_step_{step:06d}.pt")
            
            step += 1
            pbar.update(1)
            
            if step >= num_steps:
                break
    
    pbar.close()
    return encoder, decoder, losses

print("✓ Training function ready!")


# ============================================================================
# Cell 10: Run Training
# ============================================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Train!
encoder, decoder, losses = train_autoencoder(
    data_root=f"{PROJECT_ROOT}/data/train",  # Not used with dummy dataset
    device=device,
    batch_size=1,
    lr=2e-4,
    num_steps=500,  # Small number for testing
    log_every=50,
    save_every=200,
    use_skip_connections=True,
    base_ch=16,
)

print("\n✓ Training complete!")


# ============================================================================
# Cell 11: Plot Training Loss
# ============================================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.xlabel('Step')
plt.ylabel('L1 Loss')
plt.title('Training Loss')
plt.grid(True)
plt.yscale('log')
plt.show()


# ============================================================================
# Cell 12: Visualize Reconstruction
# ============================================================================

import matplotlib.pyplot as plt

# Test reconstruction
encoder.eval()
decoder.eval()

with torch.no_grad():
    # Get a test sample
    test_volume = next(iter(DataLoader(
        DummyMRIDataset(num_samples=1, volume_size=(64, 64, 64)),
        batch_size=1
    ))).to(device)
    
    # Reconstruct
    latent, skip1 = encoder(test_volume)
    reconstruction = decoder(latent, skip1)
    
    # Move to CPU for visualization
    original = test_volume[0, 0].cpu().numpy()
    recon = reconstruction[0, 0].cpu().numpy()
    
    # Plot middle slices
    mid_slice = original.shape[0] // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original[mid_slice], cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(recon[mid_slice], cmap='gray')
    axes[1].set_title('Reconstruction')
    axes[1].axis('off')
    
    diff = np.abs(original[mid_slice] - recon[mid_slice])
    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title('Difference')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Reconstruction error: {F.l1_loss(reconstruction, test_volume).item():.4e}")


# ============================================================================
# Cell 13: Load Checkpoint and Continue Training
# ============================================================================

# Load a saved checkpoint
checkpoint_path = f"{PROJECT_ROOT}/checkpoints/ae_step_000200.pt"

encoder = ShapedEncoder3D(in_ch=1, base_ch=16).to(device)
decoder = ShapedDecoder3D(out_ch=1, base_ch=16).to(device)
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()), 
    lr=2e-4
)

step = load_checkpoint(checkpoint_path, encoder, decoder, optimizer, device)

print(f"Resumed from step {step}, ready to continue training!")
