import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.encoder import ShapedEncoder3D
from models.eps_unet import ConditionalEpsUNet3D
from diffusion.schedules import DiffusionSchedule
from data.dataset import MRIDataset
from utils.checkpoint import load_checkpoint, save_checkpoint


def train_diffusion(
    data_root,
    ae_ckpt,
    device="cuda",
    batch_size=1,
    lr=2e-4,
    num_steps=50000,
    log_every=100,
    save_every=2000,
):
    # -------------------------
    # Dataset
    # -------------------------
    dataset = MRIDataset(data_root)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # -------------------------
    # Load autoencoder
    # -------------------------
    encoder = ShapedEncoder3D().to(device)
    decoder = Decoder3D().to(device)

    ckpt = torch.load(ae_ckpt, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])

    encoder.eval()
    decoder.eval()

    for p in encoder.parameters():
        p.requires_grad = False
    for p in decoder.parameters():
        p.requires_grad = False

    # -------------------------
    # Diffusion model
    # -------------------------
    eps_model = ConditionalEpsUNet3D(
        z_ch=encoder.latent_dim,
        cond_ch=encoder.latent_dim
    ).to(device)

    optimizer = torch.optim.Adam(eps_model.parameters(), lr=lr)

    schedule = DiffusionSchedule(T=1000, device=device)

    step = 0
    pbar = tqdm(total=num_steps)

    while step < num_steps:
        for hr in loader:
            if step >= num_steps:
                break

            hr = hr.to(device)

            # create LR version
            lr = F.avg_pool3d(hr, kernel_size=2)

            # encode
            with torch.no_grad():
                z_hr = encoder(hr)
                z_lr = encoder(lr)

            # upsample conditioning
            z_cond = F.interpolate(
                z_lr, size=z_hr.shape[2:], mode="trilinear", align_corners=False
            )

            # sample timestep
            # -------------------------
            # timestep sampling (biased for partial diffusion)
            # -------------------------
            B = z_hr.shape[0]
            
            u = torch.rand(B, device=device)
            t = (u ** 2 * schedule.T).long()
            t = torch.clamp(t, 0, schedule.T - 1)
            
            # -------------------------
            # forward diffusion
            # -------------------------
            noise = torch.randn_like(z_hr)
            alpha_bar = schedule.alpha_bars[t].view(B, 1, 1, 1, 1)
            
            z_t = torch.sqrt(alpha_bar) * z_hr + torch.sqrt(1 - alpha_bar) * noise
            
            # -------------------------
            # noise conditioning too (key for partdiff stability)
            # -------------------------
            cond_noise = torch.randn_like(z_cond)
            z_cond_t = (
                torch.sqrt(alpha_bar) * z_cond +
                torch.sqrt(1 - alpha_bar) * cond_noise
            )
            
            # -------------------------
            # predict noise
            # -------------------------
            pred = eps_model(z_t, t, z_cond_t)
            
            loss = F.mse_loss(pred, noise)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % log_every == 0:
                print(f"[DIFF] step {step:06d} | loss {loss.item():.4e}")

            if step % save_every == 0 and step > 0:
                save_checkpoint(
                    {
                        "eps_model": eps_model.state_dict(),
                        "step": step,
                    },
                    f"checkpoints/diffusion_step_{step}.pt",
                )

            step += 1
            pbar.update(1)

    pbar.close()


if __name__ == "__main__":
    train_diffusion(
        data_root="data/",
        ae_ckpt="checkpoints/ae_step_20000.pt"
    )

