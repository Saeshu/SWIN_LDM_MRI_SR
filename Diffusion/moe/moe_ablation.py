"""
============================================================
moe_ablation.py

Functional expert ablation.

Author:
    Diffusion Framework
============================================================
"""

from typing import Dict

import torch
import torch.nn.functional as F


@torch.no_grad()
def expert_ablation(

    unet,

    z,

    t,

    cond,

    w_e2=None,

    alpha=1.0,

    num_experts=3,

):

    #########################################################
    # Reference prediction
    #########################################################

    reference = unet(

        z,

        t,

        cond,

        w_e2,

        alpha,

        expert_mask=None,

    )

    results = {}

    #########################################################
    # Remove one expert
    #########################################################

    for expert in range(num_experts):

        mask = [1] * num_experts

        mask[expert] = 0

        pred = unet(

            z,

            t,

            cond,

            w_e2,

            alpha,

            expert_mask=mask,

        )

        results[f"expert_{expert}"] = {

            "l1":

                F.l1_loss(

                    pred,

                    reference,

                ).item(),

            "mse":

                F.mse_loss(

                    pred,

                    reference,

                ).item(),

            "cosine":

                F.cosine_similarity(

                    pred.flatten(1),

                    reference.flatten(1),

                    dim=1,

                ).mean().item(),

        }

    return results
