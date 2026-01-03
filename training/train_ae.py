import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.ShapedEncoder3D import ShapedEncoder3D
from models.Decoder import Decoder3D
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
):
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
    # Models
    # -------------------------
    encoder = ShapedEncoder3D().to(device)
    decoder = Decoder3D(in_ch=32).to(device)

    encoder.train()
    decoder.train()

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    step = 0
    pbar = tqdm(total=num_steps)

    while step < num_steps:
        for hr in loader:
            hr = hr.to(device)
            z = encoder(hr)
            recon = decoder(z)
            loss = F.l1_loss(recon, hr)

            if step % log_every == 0:
                print(f"[AE] step {step:06d} | loss {loss.item():.4e}")

            if step % save_every == 0 and step > 0:
                save_checkpoint(
                    {
                        "encoder": encoder.state_dict(),
                        "decoder": decoder.state_dict(),
                        "step": step,
                    },
                    f"checkpoints/ae_step_{step}.pt",
                )

            step += 1
            pbar.update(1)

    pbar.close()
  

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_autoencoder(
        data_root="/content/drive/MyDrive/mri_dataset/train",
        device=device,
    )

