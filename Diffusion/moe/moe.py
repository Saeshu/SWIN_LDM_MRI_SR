"""
============================================================
moe.py

Main interface for Mixture-of-Experts analysis.

This module orchestrates

    - Gate statistics
    - Routing entropy
    - Expert ablation

It performs NO analysis itself.

Author:
    Diffusion Framework
============================================================
"""

from typing import Dict

import torch

from Diffusion.moe.moe_stats import (
    create_gate_history,
    gate_statistics,
    append_gate_statistics,
    finalize_gate_history,
)

from Diffusion.moe.moe_entropy import (
    create_entropy_history,
    entropy_statistics,
    append_entropy_statistics,
    finalize_entropy_history,
)

from Diffusion.moe.moe_ablation import (
    expert_ablation,
)


# ----------------------------------------------------------
# Main API
# ----------------------------------------------------------

@torch.no_grad()
def analyze_moe(

    unet,

    z,

    t,

    cond,

    w_e2=None,

    alpha=1.0,

    num_experts=3,

):
    """
    Analyze MoE routing for a single batch.

    Parameters
    ----------
    z : [B,C,D,H,W]

    t : [B]

    cond : conditioning latent

    Returns
    -------
    Dictionary containing

        gate_stats

        entropy

        ablation
    """

    #########################################################
    # Forward once to obtain routing gates
    #########################################################

    _, gates = unet(

        z=z,

        t=t,

        cond=cond,

        w_e2=w_e2,

        alpha=alpha,

        return_gates=True,

    )

    #########################################################
    # Gate statistics
    #########################################################

    gate_history = create_gate_history()

    gate_stats = gate_statistics(gates)

    append_gate_statistics(

        gate_history,

        gate_stats,

    )

    gate_history = finalize_gate_history(

        gate_history

    )

    #########################################################
    # Entropy
    #########################################################

    entropy_history = create_entropy_history()

    entropy_stats = entropy_statistics(

        gates

    )

    append_entropy_statistics(

        entropy_history,

        entropy_stats,

    )

    entropy_history = finalize_entropy_history(

        entropy_history

    )

    #########################################################
    # Functional ablation
    #########################################################

    ablation = expert_ablation(

        unet=unet,

        z=z,

        t=t,

        cond=cond,

        w_e2=w_e2,

        alpha=alpha,

        num_experts=num_experts,

    )

    #########################################################
    # Return everything
    #########################################################

    return {

        "gates": gates.cpu(),

        "gate_statistics": gate_history,

        "entropy": entropy_history,

        "ablation": ablation,

    }
