"""
routing_analysis.py

Analysis suite for spatial specialization of adaptive decoder experts.

Expected model interface:
    z, w_enc = ae.encode(x)
    out, w_dec = ae.decode(z, return_weights=True)

Expected:
    x      : [B, C, D, H, W]
    w_dec  : [B, K, D_r, H_r, W_r]

This file analyzes the decoder routing weights and produces:
    1. Global expert utilization
    2. Routing entropy distribution + entropy map
    3. Pairwise routing-map correlation
    4. Routing weight vs. image gradient
    5. In-plane vs. through-plane gradient correlations
    6. Winner-take-all expert assignment + winner distribution
    7. Routing weight vs. local reconstruction error
    8. Weighted mean reconstruction error by expert
    9. Summary statistics saved to CSV/NPZ

Usage in your notebook:

    from routing_analysis import run_all_experiments

    results = run_all_experiments(
        ae,
        x,
        output_dir="routing_analysis"
    )

The script deliberately does not assume your model's exact class definitions.
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ============================================================
# Basic utilities
# ============================================================

def _to_cpu(t):
    if t is None:
        return None
    return t.detach().cpu()


def _safe_corr(a, b):
    """Pearson correlation for two 1D tensors."""
    a = a.float().flatten()
    b = b.float().flatten()

    a = a - a.mean()
    b = b - b.mean()

    denom = torch.sqrt(
        (a * a).sum() * (b * b).sum()
    )

    if denom < 1e-12:
        return 0.0

    return ((a * b).sum() / denom).item()


def _savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================
# Model inference
# ============================================================

@torch.no_grad()
def get_decoder_routing(ae, x):
    """
    Runs one sample through the AE and obtains decoder routing.

    Returns:
        dict containing:
            z
            w_enc
            w_dec
            output
            input
    """

    was_training = ae.training
    ae.eval()

    device = next(ae.parameters()).device
    x_in = x[:1].to(device)

    z, w_enc = ae.encode(x_in)
    out, w_dec = ae.decode(
        z,
        return_weights=True
    )

    result = {
        "input": x_in.detach().cpu(),
        "z": _to_cpu(z),
        "w_enc": _to_cpu(w_enc),
        "w_dec": _to_cpu(w_dec),
        "output": _to_cpu(out),
    }

    if was_training:
        ae.train()

    return result


# ============================================================
# 1. Global expert utilization
# ============================================================

def compute_expert_utilization(w_dec):
    """
    Returns mean routing weight for each expert.
    """
    w_dec = _to_cpu(w_dec)
    return w_dec.mean(dim=(0, 2, 3, 4)).numpy()


def plot_expert_utilization(w_dec, save_path=None):
    usage = compute_expert_utilization(w_dec)
    K = len(usage)
    uniform = 1.0 / K

    plt.figure(figsize=(7, 4))

    plt.bar(
        np.arange(K),
        usage
    )

    plt.axhline(
        uniform,
        linestyle="--",
        label=f"Uniform = {uniform:.2f}"
    )

    for k, value in enumerate(usage):
        plt.text(
            k,
            value + 0.01,
            f"{value:.3f}",
            ha="center"
        )

    plt.xticks(
        np.arange(K),
        [f"K{k}" for k in range(K)]
    )

    plt.ylabel("Mean routing weight")
    plt.xlabel("Expert")
    plt.title("Global Expert Utilization")
    plt.legend()

    if save_path:
        _savefig(save_path)
    else:
        plt.show()

    return usage


# ============================================================
# 2. Routing entropy
# ============================================================

def compute_entropy(w_dec):
    """
    Computes voxel-wise routing entropy.

    Returns:
        entropy:            [B, D, H, W]
        normalized_entropy: [B, D, H, W]

    Normalization:
        H / log(K)
    """
    w_dec = _to_cpu(w_dec)

    eps = 1e-8
    K = w_dec.shape[1]

    entropy = -(
        w_dec * torch.log(w_dec + eps)
    ).sum(dim=1)

    normalized_entropy = entropy / np.log(K)

    return entropy, normalized_entropy


def entropy_statistics(w_dec):
    entropy, normalized = compute_entropy(w_dec)

    e = entropy.flatten().numpy()
    en = normalized.flatten().numpy()

    stats = {
        "mean_entropy": float(e.mean()),
        "max_entropy": float(np.log(w_dec.shape[1])),
        "mean_normalized_entropy": float(en.mean()),
        "p10_normalized_entropy": float(np.percentile(en, 10)),
        "p25_normalized_entropy": float(np.percentile(en, 25)),
        "median_normalized_entropy": float(np.percentile(en, 50)),
        "p75_normalized_entropy": float(np.percentile(en, 75)),
        "p90_normalized_entropy": float(np.percentile(en, 90)),
    }

    return stats


def plot_entropy_distribution(w_dec, save_path=None):
    entropy, normalized = compute_entropy(w_dec)

    en = normalized.flatten().numpy()

    plt.figure(figsize=(7, 4))

    plt.hist(
        en,
        bins=50,
        density=True
    )

    mean_value = en.mean()

    plt.axvline(
        mean_value,
        linestyle="--",
        label=f"Mean = {mean_value:.3f}"
    )

    plt.xlabel("Normalized routing entropy")
    plt.ylabel("Density")
    plt.title("Distribution of Routing Entropy")
    plt.legend()

    if save_path:
        _savefig(save_path)
    else:
        plt.show()

    stats = entropy_statistics(w_dec)

    print("\nRouting entropy statistics")
    print("-" * 40)

    for key, value in stats.items():
        print(f"{key}: {value:.6f}")

    return stats


def plot_entropy_map(w_dec, slice_idx=None, save_path=None):
    _, normalized = compute_entropy(w_dec)

    if slice_idx is None:
        slice_idx = normalized.shape[1] // 2

    plt.figure(figsize=(5, 4))

    plt.imshow(
        normalized[0, slice_idx].numpy(),
        cmap="viridis",
        vmin=0,
        vmax=1
    )

    plt.colorbar(label="Normalized entropy")
    plt.title("Routing Entropy")
    plt.axis("off")

    if save_path:
        _savefig(save_path)
    else:
        plt.show()


# ============================================================
# 3. Pairwise routing-map correlation
# ============================================================

def compute_routing_correlation(w_dec):
    """
    Computes Pearson correlation between spatial routing maps.

    Important:
    Because softmax weights sum to one at every voxel,
    some negative correlations are structurally induced.
    """
    w_dec = _to_cpu(w_dec)

    # Average across batch if more than one sample is supplied.
    w = w_dec.mean(dim=0)

    K = w.shape[0]
    w = w.reshape(K, -1)

    w = w - w.mean(dim=1, keepdim=True)

    w = w / (
        w.norm(dim=1, keepdim=True) + 1e-8
    )

    corr = w @ w.T

    return corr.numpy()


def plot_routing_correlation(w_dec, save_path=None):
    corr = compute_routing_correlation(w_dec)

    K = corr.shape[0]

    plt.figure(figsize=(6, 5))

    plt.imshow(
        corr,
        vmin=-1,
        vmax=1,
        cmap="coolwarm"
    )

    plt.colorbar(label="Pearson correlation")

    plt.xticks(
        range(K),
        [f"K{k}" for k in range(K)]
    )

    plt.yticks(
        range(K),
        [f"K{k}" for k in range(K)]
    )

    for i in range(K):
        for j in range(K):
            plt.text(
                j,
                i,
                f"{corr[i, j]:.2f}",
                ha="center",
                va="center"
            )

    plt.title("Pairwise Correlation of Routing Maps")

    if save_path:
        _savefig(save_path)
    else:
        plt.show()

    print("\nRouting-map correlation matrix")
    print(corr)

    return corr


# ============================================================
# 4. Image gradient
# ============================================================

def compute_gradient_components(x):
    """
    Computes forward-difference gradients.

    x:
        [B, C, D, H, W]

    Returns:
        dx, dy, dz with same shape as x.
    """

    x = x.float()

    dx = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
    dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
    dz = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]

    dx = F.pad(
        dx,
        (0, 1, 0, 0, 0, 0)
    )

    dy = F.pad(
        dy,
        (0, 0, 0, 1, 0, 0)
    )

    dz = F.pad(
        dz,
        (0, 0, 0, 0, 0, 1)
    )

    return dx, dy, dz


def compute_gradient_magnitude(x):
    dx, dy, dz = compute_gradient_components(x)

    return torch.sqrt(
        dx ** 2 +
        dy ** 2 +
        dz ** 2 +
        1e-8
    )


def gradient_at_routing_resolution(x, w_dec):
    """
    Downsamples image gradient to the routing resolution.

    'area' is used because this is a spatial aggregation/downsampling
    rather than interpolation of an image for display.
    """

    x = _to_cpu(x)
    w_dec = _to_cpu(w_dec)

    grad = compute_gradient_magnitude(x)

    grad = F.interpolate(
        grad,
        size=w_dec.shape[2:],
        mode="area"
    )

    return grad[:, 0]


# ============================================================
# 5. Routing vs. image gradient
# ============================================================

def routing_vs_gradient(w_dec, x, n_bins=20):
    """
    Equal-population bins:
    each bin contains approximately the same number of voxels.
    """

    w_dec = _to_cpu(w_dec)
    x = _to_cpu(x[:1])

    grad = gradient_at_routing_resolution(
        x,
        w_dec
    )

    K = w_dec.shape[1]

    grad = grad.flatten()
    weights = w_dec[0].reshape(K, -1)

    order = torch.argsort(grad)
    grad_sorted = grad[order]

    bins = torch.chunk(
        torch.arange(len(grad_sorted)),
        n_bins
    )

    mean_grad = []
    mean_weights = [[] for _ in range(K)]

    for idx in bins:

        g = grad_sorted[idx]

        mean_grad.append(
            g.mean().item()
        )

        original_idx = order[idx]

        for k in range(K):
            mean_weights[k].append(
                weights[
                    k,
                    original_idx
                ].mean().item()
            )

    return (
        np.array(mean_grad),
        np.array(mean_weights)
    )


def plot_routing_vs_gradient(
    w_dec,
    x,
    n_bins=20,
    save_path=None
):

    mean_grad, mean_weights = routing_vs_gradient(
        w_dec,
        x,
        n_bins=n_bins
    )

    K = w_dec.shape[1]

    plt.figure(figsize=(7, 5))

    for k in range(K):
        plt.plot(
            mean_grad,
            mean_weights[k],
            marker="o",
            label=f"K{k}"
        )

    plt.xscale("log")

    plt.xlabel("Image gradient magnitude")
    plt.ylabel("Mean routing weight")
    plt.title("Expert Routing vs. Image Gradient")
    plt.legend()

    if save_path:
        _savefig(save_path)
    else:
        plt.show()

    return mean_grad, mean_weights


# ============================================================
# 6. Directional gradients
# ============================================================

def compute_directional_gradients(x):
    """
    Returns:

        g_xy = sqrt(dx^2 + dy^2)
        g_z  = |dz|
    """

    dx, dy, dz = compute_gradient_components(x)

    g_xy = torch.sqrt(
        dx ** 2 +
        dy ** 2 +
        1e-8
    )

    g_z = torch.abs(dz)

    return g_xy[:, 0], g_z[:, 0]


def directional_gradients_at_routing_resolution(
    x,
    w_dec
):

    x = _to_cpu(x)
    w_dec = _to_cpu(w_dec)

    g_xy, g_z = compute_directional_gradients(x)

    target_size = w_dec.shape[2:]

    g_xy = F.interpolate(
        g_xy[:, None],
        size=target_size,
        mode="area"
    )[:, 0]

    g_z = F.interpolate(
        g_z[:, None],
        size=target_size,
        mode="area"
    )[:, 0]

    return g_xy, g_z


def directional_routing_correlations(w_dec, x):

    g_xy, g_z = directional_gradients_at_routing_resolution(
        x[:1],
        w_dec
    )

    K = w_dec.shape[1]

    g_xy = g_xy.flatten()
    g_z = g_z.flatten()

    xy_corr = []
    z_corr = []

    for k in range(K):

        w = w_dec[0, k].flatten()

        xy = _safe_corr(w, g_xy)
        z = _safe_corr(w, g_z)

        xy_corr.append(xy)
        z_corr.append(z)

    xy_corr = np.array(xy_corr)
    z_corr = np.array(z_corr)

    print("\nCorrelation with in-plane gradient")
    print("-" * 40)

    for k, value in enumerate(xy_corr):
        print(f"K{k}: {value:+.4f}")

    print("\nCorrelation with through-plane gradient")
    print("-" * 40)

    for k, value in enumerate(z_corr):
        print(f"K{k}: {value:+.4f}")

    return xy_corr, z_corr


# ============================================================
# 7. Directional selectivity ratio
# ============================================================

def routing_vs_directional_ratio(
    w_dec,
    x,
    n_bins=20
):
    """
    Computes routing as a function of:

        R = g_xy / (g_z + eps)

    This is useful for testing whether anisotropic experts
    preferentially respond to in-plane vs. through-plane structure.
    """

    g_xy, g_z = directional_gradients_at_routing_resolution(
        x[:1],
        w_dec
    )

    ratio = g_xy / (g_z + 1e-8)

    ratio = ratio.flatten()

    K = w_dec.shape[1]
    weights = w_dec[0].reshape(K, -1)

    order = torch.argsort(ratio)
    ratio_sorted = ratio[order]

    bins = torch.chunk(
        torch.arange(len(ratio_sorted)),
        n_bins
    )

    mean_ratio = []
    mean_weights = [[] for _ in range(K)]

    for idx in bins:

        mean_ratio.append(
            ratio_sorted[idx].mean().item()
        )

        original_idx = order[idx]

        for k in range(K):
            mean_weights[k].append(
                weights[
                    k,
                    original_idx
                ].mean().item()
            )

    return (
        np.array(mean_ratio),
        np.array(mean_weights)
    )


def plot_routing_vs_directional_ratio(
    w_dec,
    x,
    n_bins=20,
    save_path=None
):

    ratio, mean_weights = routing_vs_directional_ratio(
        w_dec,
        x,
        n_bins=n_bins
    )

    K = w_dec.shape[1]

    plt.figure(figsize=(7, 5))

    for k in range(K):
        plt.plot(
            ratio,
            mean_weights[k],
            marker="o",
            label=f"K{k}"
        )

    plt.xscale("log")

    plt.xlabel(
        "In-plane / through-plane gradient ratio"
    )

    plt.ylabel("Mean routing weight")

    plt.title(
        "Expert Routing vs. Directional Structure"
    )

    plt.legend()

    if save_path:
        _savefig(save_path)
    else:
        plt.show()

    return ratio, mean_weights


# ============================================================
# 8. Winner-take-all assignment
# ============================================================

def compute_winner_map(w_dec):
    return torch.argmax(
        _to_cpu(w_dec),
        dim=1
    )


def winner_statistics(w_dec):

    winner = compute_winner_map(w_dec)

    K = w_dec.shape[1]

    counts = torch.bincount(
        winner.flatten(),
        minlength=K
    )

    proportions = (
        counts.float() /
        counts.sum()
    )

    print("\nWinner distribution")
    print("-" * 40)

    for k in range(K):
        print(
            f"K{k}: {proportions[k].item():.4f}"
        )

    return proportions.numpy()


def plot_winner_map(
    w_dec,
    slice_idx=None,
    save_path=None
):

    winner = compute_winner_map(w_dec)

    if slice_idx is None:
        slice_idx = winner.shape[1] // 2

    winner_slice = winner[
        0,
        slice_idx
    ].numpy()

    K = w_dec.shape[1]

    plt.figure(figsize=(6, 5))

    plt.imshow(
        winner_slice,
        interpolation="nearest",
        cmap="tab10",
        vmin=0,
        vmax=K - 1
    )

    cbar = plt.colorbar(
        ticks=np.arange(K)
    )

    cbar.ax.set_yticklabels(
        [f"K{k}" for k in range(K)]
    )

    plt.title(
        "Winner-Take-All Expert Assignment"
    )

    plt.axis("off")

    if save_path:
        _savefig(save_path)
    else:
        plt.show()


# ============================================================
# 9. Routing vs. reconstruction error
# ============================================================

def reconstruction_error_at_routing_resolution(
    x,
    output,
    w_dec
):
    """
    Computes absolute reconstruction error and downsamples
    it to the routing resolution.
    """

    x = _to_cpu(x[:1]).float()
    output = _to_cpu(output[:1]).float()
    w_dec = _to_cpu(w_dec)

    error = torch.abs(
        output - x
    )

    error = F.interpolate(
        error,
        size=w_dec.shape[2:],
        mode="area"
    )

    return error[:, 0]


def routing_vs_reconstruction_error(
    w_dec,
    x,
    output,
    n_bins=20
):
    """
    For each expert, calculate how its routing weight changes
    with local reconstruction error.

    Also computes weighted mean error:

        sum(w_k * error) / sum(w_k)
    """

    error = reconstruction_error_at_routing_resolution(
        x,
        output,
        w_dec
    )

    error = error.flatten()

    K = w_dec.shape[1]

    weights = w_dec[0].reshape(K, -1)

    order = torch.argsort(error)
    error_sorted = error[order]

    bins = torch.chunk(
        torch.arange(len(error_sorted)),
        n_bins
    )

    mean_error = []
    mean_weights = [[] for _ in range(K)]

    for idx in bins:

        mean_error.append(
            error_sorted[idx].mean().item()
        )

        original_idx = order[idx]

        for k in range(K):
            mean_weights[k].append(
                weights[
                    k,
                    original_idx
                ].mean().item()
            )

    weighted_error = (
        (weights * error[None, :]).sum(dim=1)
        /
        (weights.sum(dim=1) + 1e-8)
    )

    return (
        np.array(mean_error),
        np.array(mean_weights),
        weighted_error.numpy()
    )


def plot_routing_vs_reconstruction_error(
    w_dec,
    x,
    output,
    n_bins=20,
    save_path=None
):

    mean_error, mean_weights, weighted_error = (
        routing_vs_reconstruction_error(
            w_dec,
            x,
            output,
            n_bins=n_bins
        )
    )

    K = w_dec.shape[1]

    plt.figure(figsize=(7, 5))

    for k in range(K):

        plt.plot(
            mean_error,
            mean_weights[k],
            marker="o",
            label=f"K{k}"
        )

    plt.xscale("log")

    plt.xlabel("Local absolute reconstruction error")
    plt.ylabel("Mean routing weight")

    plt.title(
        "Expert Routing vs. Reconstruction Error"
    )

    plt.legend()

    if save_path:
        _savefig(save_path)
    else:
        plt.show()

    return mean_error, mean_weights, weighted_error


def plot_weighted_error_by_expert(
    weighted_error,
    save_path=None
):

    K = len(weighted_error)

    plt.figure(figsize=(7, 4))

    plt.bar(
        np.arange(K),
        weighted_error
    )

    for k, value in enumerate(weighted_error):
        plt.text(
            k,
            value * 1.1,
            f"{value:.3g}",
            ha="center"
        )

    plt.yscale("log")

    plt.xticks(
        np.arange(K),
        [f"K{k}" for k in range(K)]
    )

    plt.ylabel(
        "Routing-weighted mean absolute error"
    )

    plt.xlabel("Expert")

    plt.title(
        "Weighted Reconstruction Error by Expert"
    )

    if save_path:
        _savefig(save_path)
    else:
        plt.show()


# ============================================================
# 10. Expert spatial maps
# ============================================================

def plot_routing_maps(
    w_dec,
    slice_idx=None,
    save_path=None
):

    w_dec = _to_cpu(w_dec)

    if slice_idx is None:
        slice_idx = w_dec.shape[2] // 2

    K = w_dec.shape[1]

    plt.figure(
        figsize=(4 * K, 4)
    )

    for k in range(K):

        plt.subplot(
            1,
            K,
            k + 1
        )

        plt.imshow(
            w_dec[
                0,
                k,
                slice_idx
            ].numpy(),
            cmap="viridis"
        )

        plt.title(
            f"Decoder Kernel {k}"
        )

        plt.colorbar()

    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight"
        )
        plt.close()
    else:
        plt.show()


# ============================================================
# 11. Save numerical results
# ============================================================

def save_results(
    output_dir,
    usage,
    entropy_stats,
    correlation,
    xy_corr,
    z_corr,
    winner_distribution,
    weighted_error,
    gradient_bins=None,
    gradient_weights=None,
    directional_ratio_bins=None,
    directional_ratio_weights=None,
    error_bins=None,
    error_weights=None,
):

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    K = len(usage)

    # Expert summary
    expert_summary = np.column_stack([
        np.arange(K),
        usage,
        winner_distribution,
        xy_corr,
        z_corr,
        weighted_error,
    ])

    np.savetxt(
        output_dir / "expert_summary.csv",
        expert_summary,
        delimiter=",",
        header=(
            "expert,mean_weight,winner_fraction,"
            "xy_gradient_corr,z_gradient_corr,"
            "weighted_mean_abs_error"
        ),
        comments=""
    )

    # Entropy
    with open(
        output_dir / "entropy_statistics.txt",
        "w"
    ) as f:
        for key, value in entropy_stats.items():
            f.write(
                f"{key}: {value:.8f}\n"
            )

    # Correlation matrix
    np.savetxt(
        output_dir / "routing_correlation.csv",
        correlation,
        delimiter=","
    )

    # Gradient relationship
    if gradient_bins is not None:
        np.savez(
            output_dir / "routing_vs_gradient.npz",
            gradient=gradient_bins,
            weights=gradient_weights
        )

    # Directional relationship
    if directional_ratio_bins is not None:
        np.savez(
            output_dir / "routing_vs_directional_ratio.npz",
            ratio=directional_ratio_bins,
            weights=directional_ratio_weights
        )

    # Error relationship
    if error_bins is not None:
        np.savez(
            output_dir / "routing_vs_error.npz",
            error=error_bins,
            weights=error_weights
        )


# ============================================================
# 12. Run everything
# ============================================================

def run_all_experiments(
    ae,
    x,
    output_dir="routing_analysis",
    slice_idx=None,
    n_bins=20,
):
    """
    Main entry point.

    Runs all analyses that can be obtained from:
        ae.encode(...)
        ae.decode(..., return_weights=True)

    Returns a dictionary containing all numerical results.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("ADAPTIVE DECODER ROUTING ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Run model once
    # --------------------------------------------------------

    result = get_decoder_routing(
        ae,
        x
    )

    x_cpu = result["input"]
    w_dec = result["w_dec"]
    output = result["output"]

    print("\nShapes")
    print("-" * 40)
    print("Input :", tuple(x_cpu.shape))
    print("Latent:", tuple(result["z"].shape))
    print("w_dec :", tuple(w_dec.shape))
    print("Output:", tuple(output.shape))

    K = w_dec.shape[1]

    # --------------------------------------------------------
    # Routing maps
    # --------------------------------------------------------

    plot_routing_maps(
        w_dec,
        slice_idx=slice_idx,
        save_path=output_dir / "01_routing_maps.png"
    )

    # --------------------------------------------------------
    # 1. Utilization
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("1. GLOBAL EXPERT UTILIZATION")
    print("=" * 70)

    usage = plot_expert_utilization(
        w_dec,
        save_path=output_dir / "02_expert_utilization.png"
    )

    print("\nExpert mean weights:")
    for k, value in enumerate(usage):
        print(f"K{k}: {value:.6f}")

    # --------------------------------------------------------
    # 2. Entropy
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("2. ROUTING ENTROPY")
    print("=" * 70)

    entropy_stats = plot_entropy_distribution(
        w_dec,
        save_path=output_dir / "03_entropy_distribution.png"
    )

    plot_entropy_map(
        w_dec,
        slice_idx=slice_idx,
        save_path=output_dir / "04_entropy_map.png"
    )

    # --------------------------------------------------------
    # 3. Correlation
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("3. PAIRWISE ROUTING CORRELATION")
    print("=" * 70)

    correlation = plot_routing_correlation(
        w_dec,
        save_path=output_dir / "05_routing_correlation.png"
    )

    # --------------------------------------------------------
    # 4. Gradient
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("4. ROUTING VS. IMAGE GRADIENT")
    print("=" * 70)

    gradient_bins, gradient_weights = plot_routing_vs_gradient(
        w_dec,
        x_cpu,
        n_bins=n_bins,
        save_path=output_dir / "06_routing_vs_gradient.png"
    )

    # --------------------------------------------------------
    # 5. Directional gradients
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("5. DIRECTIONAL GRADIENT CORRELATIONS")
    print("=" * 70)

    xy_corr, z_corr = directional_routing_correlations(
        w_dec,
        x_cpu
    )

    # --------------------------------------------------------
    # 6. Directional ratio
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("6. ROUTING VS. DIRECTIONAL STRUCTURE")
    print("=" * 70)

    directional_ratio_bins, directional_ratio_weights = (
        plot_routing_vs_directional_ratio(
            w_dec,
            x_cpu,
            n_bins=n_bins,
            save_path=output_dir / "07_routing_vs_directional_ratio.png"
        )
    )

    # --------------------------------------------------------
    # 7. Winner assignment
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("7. WINNER-TAKE-ALL ROUTING")
    print("=" * 70)

    winner_distribution = winner_statistics(
        w_dec
    )

    plot_winner_map(
        w_dec,
        slice_idx=slice_idx,
        save_path=output_dir / "08_winner_map.png"
    )

    # --------------------------------------------------------
    # 8. Reconstruction error
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("8. ROUTING VS. RECONSTRUCTION ERROR")
    print("=" * 70)

    (
        error_bins,
        error_weights,
        weighted_error
    ) = plot_routing_vs_reconstruction_error(
        w_dec,
        x_cpu,
        output,
        n_bins=n_bins,
        save_path=output_dir / "09_routing_vs_error.png"
    )

    plot_weighted_error_by_expert(
        weighted_error,
        save_path=output_dir / "10_weighted_error_by_expert.png"
    )

    print("\nRouting-weighted mean absolute error:")
    for k, value in enumerate(weighted_error):
        print(f"K{k}: {value:.8f}")

    # --------------------------------------------------------
    # Save numerical results
    # --------------------------------------------------------

    save_results(
        output_dir=output_dir,
        usage=usage,
        entropy_stats=entropy_stats,
        correlation=correlation,
        xy_corr=xy_corr,
        z_corr=z_corr,
        winner_distribution=winner_distribution,
        weighted_error=weighted_error,
        gradient_bins=gradient_bins,
        gradient_weights=gradient_weights,
        directional_ratio_bins=directional_ratio_bins,
        directional_ratio_weights=directional_ratio_weights,
        error_bins=error_bins,
        error_weights=error_weights,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {output_dir.resolve()}")

    print("\nGenerated files:")
    for path in sorted(output_dir.iterdir()):
        print(f"  {path.name}")

    return {
        "input": x_cpu,
        "output": output,
        "z": result["z"],
        "w_enc": result["w_enc"],
        "w_dec": w_dec,
        "expert_usage": usage,
        "entropy_stats": entropy_stats,
        "routing_correlation": correlation,
        "gradient_bins": gradient_bins,
        "gradient_weights": gradient_weights,
        "xy_gradient_corr": xy_corr,
        "z_gradient_corr": z_corr,
        "directional_ratio_bins": directional_ratio_bins,
        "directional_ratio_weights": directional_ratio_weights,
        "winner_distribution": winner_distribution,
        "error_bins": error_bins,
        "error_weights": error_weights,
        "weighted_error": weighted_error,
    }


# ============================================================
# Optional command-line example
# ============================================================

if __name__ == "__main__":

    print(
        "\nThis module requires your trained `ae` model and an input tensor `x`."
    )

    print(
        "\nUse it from your notebook with:"
    )

    print(
        "\n"
        "from routing_analysis import run_all_experiments\n"
        "results = run_all_experiments(ae, x)\n"
    )

    print(
        "\nThe model is expected to implement:"
    )

    print(
        "    z, w_enc = ae.encode(x)"
    )

    print(
        "    out, w_dec = ae.decode(z, return_weights=True)"
    )
