"""
============================================================
moe_stats.py

Statistics for Mixture-of-Experts routing.

Operates only on gate tensors.

Input:
    gates : [B,E]

Author:
    Diffusion Framework
============================================================
"""

from typing import Dict

import torch


# ----------------------------------------------------------
# Create history
# ----------------------------------------------------------

def create_gate_history():

    return {

        "mean": [],

        "std": [],

        "variance": [],

        "dominance": [],

        "effective_experts": [],

        "winner_frequency": [],

    }


# ----------------------------------------------------------
# Effective experts
# ----------------------------------------------------------

def effective_experts(
    gates: torch.Tensor,
) -> float:
    """
    Participation ratio.

    1 = one expert

    E = uniform routing
    """

    p = gates.mean(dim=0)

    return (

        1.0 /

        torch.sum(p ** 2)

    ).item()


# ----------------------------------------------------------
# Winner frequency
# ----------------------------------------------------------

def winner_frequency(
    gates: torch.Tensor,
):

    winners = gates.argmax(dim=1)

    counts = torch.bincount(

        winners,

        minlength=gates.shape[1]

    ).float()

    counts /= counts.sum()

    return counts.cpu()


# ----------------------------------------------------------
# Gate statistics
# ----------------------------------------------------------

def gate_statistics(
    gates: torch.Tensor,
) -> Dict:

    return {

        "mean":
            gates.mean(dim=0).cpu(),

        "std":
            gates.std(dim=0).cpu(),

        "variance":
            gates.var(dim=0).cpu(),

        "dominance":
            gates.max(dim=1).values.mean().item(),

        "effective_experts":
            effective_experts(gates),

        "winner_frequency":
            winner_frequency(gates),

    }


# ----------------------------------------------------------
# Append
# ----------------------------------------------------------

def append_gate_statistics(

    history,

    stats,

):

    history["mean"].append(
        stats["mean"]
    )

    history["std"].append(
        stats["std"]
    )

    history["variance"].append(
        stats["variance"]
    )

    history["dominance"].append(
        stats["dominance"]
    )

    history["effective_experts"].append(
        stats["effective_experts"]
    )

    history["winner_frequency"].append(
        stats["winner_frequency"]
    )


# ----------------------------------------------------------
# Finalize
# ----------------------------------------------------------

def finalize_gate_history(
    history,
):

    history["mean"] = torch.stack(
        history["mean"]
    )

    history["std"] = torch.stack(
        history["std"]
    )

    history["variance"] = torch.stack(
        history["variance"]
    )

    history["dominance"] = torch.tensor(
        history["dominance"]
    )

    history["effective_experts"] = torch.tensor(
        history["effective_experts"]
    )

    history["winner_frequency"] = torch.stack(
        history["winner_frequency"]
    )

    history["timesteps"].append(timestep)

    return history
