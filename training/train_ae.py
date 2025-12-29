import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.encoder import ShapedEncoder3D
from models.decoder import Decoder3D
from data.dataset import MRIDataset
from utils.checkpoint import save_checkpoint


def train_autoencoder(
    data_root,
    device="cuda",
    batch_size=1,
    lr=2e-4,
    num_steps=20000,
    log_every=100,
    save_every=2000,
):
    # -------------------------
    # Dataset
    # -------------------------
    dataset = MRIDataset(data_root)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # -------------------------
    # Models
    # -------------------------
    encoder = ShapedEncoder3D().to(device)
    decoder = Decoder3D().to(device)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    step = 0
    pbar = tqdm(total=num_steps)

    while step < num_steps:
        for x in loader:
            if step >= num_steps:
                break

            x = x.to(device)

            # forward
            z = encoder(x)
            recon = decoder(z)

            loss = F.l1_loss(recon, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
    train_autoencoder()
