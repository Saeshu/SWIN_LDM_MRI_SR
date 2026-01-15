import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.encoder import ShapedEncoder3D
from models.decoder import Decoder3D
from models.eps_unet import ConditionalEpsUNet3D
from diffusion.schedules import DiffusionSchedule
from data.dataset import MRIDataset
from utils.checkpoint import save_checkpoint
from utils.patches import random_patch_3d   # your patch sampler


def train_diffusion(
    data_root,
    ae_ckpt,
    device="cuda",
    lr=2e-4,
    num_steps=10000,
    log_every=100,
    save_every=2000,
    patch_size=64,
    t_min=0,
):
    # -------------------------
    # Dataset
    # -------------------------
    dataset = MRIDataset(data_root)

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
        z_ch=32,
        cond_ch=32
    ).to(device)

    optimizer = torch.optim.Adam(eps_model.parameters(), lr=lr)
    schedule = DiffusionSchedule(T=1000, device=device)

    step = 1
    pbar = tqdm(total=num_steps)

    while step < num_steps:
        idx = torch.randint(0, len(dataset), (1,)).item()
        vol = dataset[idx]          # [1, D, H, W]

        # patch training (important!)
        hr = random_patch_3d(vol, patch_size)
        hr = hr.unsqueeze(0).to(device)

        # LR version
        lr_img = F.avg_pool3d(hr, kernel_size=2)

        # encode
        with torch.no_grad():
            z_hr, _ = encoder(hr)
            z_lr, _ = encoder(lr_img)

        # upsample condition
        z_cond = F.interpolate(
            z_lr,
            size=z_hr.shape[2:],
            mode="trilinear",
            align_corners=False
        )

        # -------------------------
        # partial diffusion timestep
        # -------------------------
        B = z_hr.shape[0]
        t = torch.randint(
            low=t_min,
            high=schedule.T,
            size=(B,),
            device=device
        )

        # -------------------------
        # forward diffusion
        # -------------------------
        noise = torch.randn_like(z_hr)
        alpha_bar = schedule.alpha_bars[t].view(B, 1, 1, 1, 1)

        z_t = torch.sqrt(alpha_bar) * z_hr + torch.sqrt(1 - alpha_bar) * noise

        if step < 5000:
            z_cond_t = z_cond
        #change after first run!
        else:
            cond_noise = 0.1 * torch.randn_like(z_cond)
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
                f"checkpoints/2nd_diffusion_step_{step}.pt",
            )
        if step % 1000 == 0 and step > 0:
            t0 = torch.zeros_like(t)
            z0 = z_hr
            pred0 = eps_model(z0, t0, z_cond)
            print("Identity eps norm:", pred0.abs().mean().item())

        step += 1
        pbar.update(1)

    pbar.close()
