import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


@torch.no_grad()
def evaluate_unmasked(
    model,
    test_loader,
    device,
    data_range=1.0,
):
    model.eval()

    mse_values = []
    psnr_values = []
    ssim_values = []

    pbar = tqdm(test_loader, desc="Evaluation")

    for batch in pbar:

        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch

        x = x.to(device, non_blocking=True)

        output = model(x)

        # Handle models that return multiple outputs.
        if isinstance(output, (tuple, list)):
            reconstruction = output[1]
        else:
            reconstruction = output

        x_cpu = x.detach().cpu()
        recon_cpu = reconstruction.detach().cpu()

        for i in range(x_cpu.shape[0]):

            target = x_cpu[i]
            pred = recon_cpu[i]

            target_np = target.squeeze(0).numpy()
            pred_np = pred.squeeze(0).numpy()

            mse = np.mean(
                (pred_np - target_np) ** 2
            )

            if mse == 0:
                psnr = float("inf")
            else:
                psnr = 10 * np.log10(
                    (data_range ** 2) / mse
                )

            ssim_value = ssim(
                target_np,
                pred_np,
                data_range=data_range,
                channel_axis=None,
            )

            mse_values.append(mse)
            psnr_values.append(psnr)
            ssim_values.append(ssim_value)

    results = {
        "mse": np.array(mse_values),
        "psnr": np.array(psnr_values),
        "ssim": np.array(ssim_values),
    }

    summary = {
        "mse_mean": np.mean(results["mse"]),
        "mse_std": np.std(results["mse"]),
        "mse_median": np.median(results["mse"]),

        "psnr_mean": np.mean(results["psnr"]),
        "psnr_std": np.std(results["psnr"]),
        "psnr_median": np.median(results["psnr"]),

        "ssim_mean": np.mean(results["ssim"]),
        "ssim_std": np.std(results["ssim"]),
        "ssim_median": np.median(results["ssim"]),
    }

    return results, summary
