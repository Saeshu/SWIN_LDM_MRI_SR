# models/ae.py

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .ShapedEncoder3D import (
    AnisotropicSwinBlock,
    SpatialDownsample3D,
)

from .Decoder import (
    DecoderBlock,
    OutputRefinementHead,
)


# ============================================================
# Default expert configurations
# ============================================================

DEFAULT_ENCODER_EXPERTS = [
    "low",
    "point",
    "spatial",
    "depth",
    "identity_like",
]

DEFAULT_DECODER_EXPERTS = [
    "low",
    "high",
    "spatial",
    "point",
    "depth",
]


class AutoEncoder(nn.Module):

    def __init__(
        self,
        encoder_experts=None,
        decoder_experts=None,
    ):
        super().__init__()

        # --------------------------------------------------------
        # Expert configuration
        # --------------------------------------------------------

        if encoder_experts is None:
            encoder_experts = DEFAULT_ENCODER_EXPERTS

        if decoder_experts is None:
            decoder_experts = DEFAULT_DECODER_EXPERTS

        self.encoder_experts = list(encoder_experts)
        self.decoder_experts = list(decoder_experts)

        # --------------------------------------------------------
        # WE2 FiLM
        # --------------------------------------------------------

        self.we2_film = WE2FiLM()

        # ========================================================
        # ENCODER
        # ========================================================

        # --------------------------------------------------------
        # Encoder block 0
        #
        # No attention:
        # uniform 1/K mixture of selected experts.
        # --------------------------------------------------------

        self.enc0 = AnisotropicSwinBlock(
            1,
            32,
            use_attention=False,
            experts=self.encoder_experts,
        )

        self.down0 = SpatialDownsample3D()

        # --------------------------------------------------------
        # Encoder block 1
        #
        # No attention:
        # uniform 1/K mixture of selected experts.
        # --------------------------------------------------------

        self.enc1 = AnisotropicSwinBlock(
            32,
            64,
            use_attention=False,
            experts=self.encoder_experts,
        )

        self.down1 = SpatialDownsample3D()

        # --------------------------------------------------------
        # Encoder block 2
        #
        # Attention-based spatial routing.
        # --------------------------------------------------------

        self.enc2 = AnisotropicSwinBlock(
            64,
            128,
            use_attention=True,
            shift=False,
            experts=self.encoder_experts,
        )

        # --------------------------------------------------------
        # Encoder block 3
        #
        # Attention-based spatial routing.
        # --------------------------------------------------------

        self.enc3 = AnisotropicSwinBlock(
            128,
            256,
            use_attention=True,
            shift=False,
            experts=self.encoder_experts,
        )

        # ========================================================
        # DECODER
        # ========================================================

        # --------------------------------------------------------
        # Decoder block 2
        #
        # Adaptive/uniform routing is controlled by routing_mode.
        #
        # Current model:
        #     routing_mode="uniform"
        #     use_routing=True
        #
        # Therefore this block uses all selected decoder experts
        # with uniform 1/K weights.
        # --------------------------------------------------------

        self.dec2 = DecoderBlock(
            256,
            128,
            experts=self.decoder_experts,
            routing_mode="uniform",
            upsample=True,
            use_routing=True,
        )

        # --------------------------------------------------------
        # Decoder block 1
        #
        # No routing:
        # uniform average of selected experts.
        # --------------------------------------------------------

        self.dec1 = DecoderBlock(
            128,
            64,
            experts=self.decoder_experts,
            upsample=True,
            use_routing=False,
        )

        # --------------------------------------------------------
        # Decoder block 0
        #
        # No routing:
        # uniform average of selected experts.
        # --------------------------------------------------------

        self.dec0 = DecoderBlock(
            64,
            32,
            experts=self.decoder_experts,
            upsample=False,
            use_routing=False,
        )

        # --------------------------------------------------------
        # Output refinement
        # --------------------------------------------------------

        self.out = OutputRefinementHead(
            32,
            out_ch=1,
        )

    # ============================================================
    # ENCODE
    # ============================================================

    def encode(
        self,
        x,
        return_features=False,
    ):

        features = {}

        # --------------------------------------------------------
        # Encoder block 0
        # --------------------------------------------------------

        x = self.enc0(x)

        if return_features:
            features["enc0"] = x

        x = self.down0(x)

        # --------------------------------------------------------
        # Encoder block 1
        # --------------------------------------------------------

        x = self.enc1(x)

        if return_features:
            features["enc1"] = x

        x = self.down1(x)

        # --------------------------------------------------------
        # Encoder block 2
        #
        # This is an attention-routed encoder block.
        # --------------------------------------------------------

        out = self.enc2(
            x,
            return_weights=True,
        )

        if isinstance(out, tuple):
            x, w_E2 = out
        else:
            x = out
            w_E2 = None

        if return_features:
            features["enc2"] = x

        # --------------------------------------------------------
        # Encoder block 3
        #
        # Also attention-routed, but routing weights are not
        # currently returned.
        # --------------------------------------------------------

        x = self.enc3(x)

        if return_features:
            features["enc3"] = x

        # --------------------------------------------------------
        # Return
        # --------------------------------------------------------

        if return_features:
            return {
                "features": features,
                "latent": x,
                "w_E2": w_E2,
            }

        return x, w_E2

    # ============================================================
    # DECODE
    # ============================================================

    def decode(
        self,
        z,
        return_weights=False,
        return_expert_features=False,
    ):

        # --------------------------------------------------------
        # Decoder block 2
        # --------------------------------------------------------

        dec2_output = self.dec2(
            z,
            return_weights=(
                return_weights or return_expert_features
            ),
            return_expert_features=return_expert_features,
        )

        # --------------------------------------------------------
        # Unpack dec2 outputs
        # --------------------------------------------------------

        if return_weights and return_expert_features:

            z, weights, expert_features = dec2_output

        elif return_weights:

            z, weights = dec2_output
            expert_features = None

        elif return_expert_features:

            z, expert_features = dec2_output
            weights = None

        else:

            # ----------------------------------------------------
            # Memory-efficient checkpointed execution
            # ----------------------------------------------------

            z = checkpoint(
                self.dec2,
                z,
                use_reentrant=False,
            )

            weights = None
            expert_features = None

        # --------------------------------------------------------
        # Decoder block 1
        # --------------------------------------------------------

        z = checkpoint(
            self.dec1,
            z,
            use_reentrant=False,
        )

        # --------------------------------------------------------
        # Decoder block 0
        # --------------------------------------------------------

        z = checkpoint(
            self.dec0,
            z,
            use_reentrant=False,
        )

        # --------------------------------------------------------
        # Output refinement
        # --------------------------------------------------------

        out = self.out(z)

        # --------------------------------------------------------
        # Return
        # --------------------------------------------------------

        if return_weights and return_expert_features:

            return (
                out,
                weights,
                expert_features,
            )

        elif return_weights:

            return (
                out,
                weights,
            )

        elif return_expert_features:

            return (
                out,
                expert_features,
            )

        return out

    # ============================================================
    # FORWARD
    # ============================================================

    def forward(
        self,
        x,
        return_weights=False,
    ):

        # --------------------------------------------------------
        # Encode
        # --------------------------------------------------------

        z, w_enc = self.encode(x)

        # --------------------------------------------------------
        # Decode
        # --------------------------------------------------------

        if return_weights:

            out, w_dec = self.decode(
                z,
                return_weights=True,
            )

            return (
                out,
                w_enc,
                w_dec,
            )

        out = self.decode(z)

        return out
