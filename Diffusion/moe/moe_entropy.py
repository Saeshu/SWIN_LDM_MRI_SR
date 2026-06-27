"""
============================================================
moe_entropy.py

Entropy-based routing analysis.

Author:
    Diffusion Framework
============================================================
"""

from typing import Dict

import math
import torch


# ----------------------------------------------------------
# Shannon entropy
# ----------------------------------------------------------

def routing_entropy(
    gates: torch.Tensor,
) -> torch.Tensor:
    """
    Parameters
    ----------
    gates : [B,E]

    Returns
    -------
    entropy : [B]
    """

    p = gates.clamp(min=1e-8)

    entropy = -(p * torch.log(p)).sum(dim=1)

    return entropy


# ----------------------------------------------------------
# Normalized entropy
# ----------------------------------------------------------

def normalized_entropy(
    gates: torch.Tensor,
) -> torch.Tensor:

    H = routing_entropy(gates)

    Hmax = math.log(gates.shape[1])

    return H / Hmax


# ----------------------------------------------------------
# Top1-Top2 margin
# ----------------------------------------------------------

def routing_margin(
    gates: torch.Tensor,
) -> torch.Tensor:

    top2 = torch.topk(
        gates,
        k=2,
        dim=1,
    ).values

    return top2[:,0] - top2[:,1]


# ----------------------------------------------------------
# Collapse ratio
# ----------------------------------------------------------

def collapse_ratio(
    gates: torch.Tensor,
    threshold=0.95,
):

    winners = gates.max(dim=1).values

    return (
        winners > threshold
    ).float().mean().item()


# ----------------------------------------------------------
# Main API
# ----------------------------------------------------------

def entropy_statistics(
    gates: torch.Tensor,
) -> Dict:

    H = routing_entropy(gates)

    Hn = normalized_entropy(gates)

    margin = routing_margin(gates)

    return {

        "entropy":

            H.mean().item(),

        "entropy_std":

            H.std().item(),

        "normalized_entropy":

            Hn.mean().item(),

        "margin":

            margin.mean().item(),

        "margin_std":

            margin.std().item(),

        "collapse_ratio":

            collapse_ratio(gates),

    }


# ----------------------------------------------------------
# History
# ----------------------------------------------------------

def create_entropy_history():

    return {

        "entropy": [],

        "entropy_std": [],

        "normalized_entropy": [],

        "margin": [],

        "margin_std": [],

        "collapse_ratio": [],

    }


# ----------------------------------------------------------
# Append
# ----------------------------------------------------------

def append_entropy_statistics(

    history,

    stats,

):

    for key in history:

        history[key].append(

            stats[key]

        )


# ----------------------------------------------------------
# Finalize
# ----------------------------------------------------------

def finalize_entropy_history(

    history,

):

    for key in history:

        history[key] = torch.tensor(

            history[key]

        )

    return history
