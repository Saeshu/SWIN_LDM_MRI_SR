import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Anisotropic convolution expert suite
# ============================================================

class AnisotropicConvSuite(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.kernels = nn.ModuleList([

            # 1. Low-pass / structure
            nn.Sequential(
                nn.AvgPool3d(
                    kernel_size=(1, 3, 3),
                    stride=1,
                    padding=(0, 1, 1)
                ),
                nn.Conv3d(
                    in_ch,
                    out_ch,
                    kernel_size=1
                )
            ),

            # 2. Pointwise
            nn.Conv3d(
                in_ch,
                out_ch,
                kernel_size=1
            ),

            # 3. Spatial
            nn.Conv3d(
                in_ch,
                out_ch,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1)
            ),

            # 4. Depth
            nn.Conv3d(
                in_ch,
                out_ch,
                kernel_size=(3, 1, 1),
                padding=(1, 0, 0)
            ),

            # 5. Identity-like
            nn.Conv3d(
                in_ch,
                out_ch,
                kernel_size=1
            )
        ])

        self.num_paths = len(self.kernels)

    def forward(self, x):
        """
        Original behavior.
        Returns all expert outputs.
        Used by attention blocks.
        """

        return [
            expert(x)
            for expert in self.kernels
        ]

    def forward_sequential(self, x, weights):

        y = self.kernels[0](x) * weights[0]
    
        for i in range(1, self.num_paths):
    
            feat = self.kernels[i](x)
    
            y = y + weights[i] * feat
    
            del feat
    
        return y

# ============================================================
# Window pooling / tokenization
# ============================================================

class WindowPool3D(nn.Module):

    def __init__(
        self,
        window_size=None,
        shift=False
    ):

        super().__init__()

        self.window_size = window_size
        self.shift = shift

    def forward(self, x):

        B, C, D, H, W = x.shape

        # ----------------------------------------------------
        # Determine window size
        # ----------------------------------------------------

        if self.window_size is None:

            target_grid = (32, 4, 4)

            wd = max(
                1,
                D // target_grid[0]
            )

            wh = max(
                3,
                H // target_grid[1]
            )

            ww = max(
                3,
                W // target_grid[2]
            )

            # Keep H/W window sizes odd
            if wh % 2 == 0:
                wh += 1

            if ww % 2 == 0:
                ww += 1

            wd = min(wd, D)
            wh = min(wh, H)
            ww = min(ww, W)

        else:

            wd, wh, ww = self.window_size

        # ----------------------------------------------------
        # Padding
        # ----------------------------------------------------

        pad_d = (wd - D % wd) % wd
        pad_h = (wh - H % wh) % wh
        pad_w = (ww - W % ww) % ww

        x = F.pad(
            x,
            (
                0, pad_w,
                0, pad_h,
                0, pad_d
            )
        )

        D_pad, H_pad, W_pad = x.shape[2:]

        # ----------------------------------------------------
        # Window partition
        # ----------------------------------------------------

        x = (
            x.unfold(2, wd, wd)
             .unfold(3, wh, wh)
             .unfold(4, ww, ww)
        )

        Nd, Nh, Nw = x.shape[2:5]

        x = x.contiguous().view(
            B,
            C,
            Nd * Nh * Nw,
            wd * wh * ww
        )

        # ----------------------------------------------------
        # Tokenization
        # ----------------------------------------------------

        mean = x.mean(dim=-1)
        std = x.std(dim=-1)

        raw_tokens = mean + 0.1 * std

        tokens = raw_tokens / (
            raw_tokens.std(
                dim=-1,
                keepdim=True
            ) + 1e-6
        )

        tokens = tokens.permute(
            0,
            2,
            1
        )

        # Only return what the attention block actually needs
        return (
            tokens,
            (Nd, Nh, Nw),
            (D_pad, H_pad, W_pad)
        )


# ============================================================
# Attention → kernel logits
# ============================================================

class KernelMixingAttention(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_kernels,
        num_heads=4
    ):

        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.proj = nn.Linear(
            embed_dim,
            num_kernels
        )

    def forward(self, tokens):

        attn_out, _ = self.attn(
            tokens,
            tokens,
            tokens
        )

        logits = self.proj(attn_out)

        return logits


# ============================================================
# Encoder block
# ============================================================

class AnisotropicSwinBlock(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch,
        window_size=None,
        use_attention=True,
        shift=False
    ):

        super().__init__()

        self.use_attention = use_attention
        self.num_kernels = 5
        self.window_size = window_size

        # ----------------------------------------------------
        # Expert suite
        # ----------------------------------------------------

        self.conv_suite = AnisotropicConvSuite(
            in_ch,
            out_ch
        )

        # ----------------------------------------------------
        # Reduced representation for routing
        # ----------------------------------------------------

        reduced_ch = max(
            1,
            in_ch // 2
        )

        self.reduce = nn.Conv3d(
            in_ch,
            reduced_ch,
            kernel_size=1
        )

        # ----------------------------------------------------
        # Routing
        # ----------------------------------------------------

        if use_attention:

            self.window_pool = WindowPool3D(
                window_size,
                shift=False
            )

            self.attn = KernelMixingAttention(
                embed_dim=reduced_ch,
                num_kernels=self.num_kernels
            )

        else:

            self.alpha = nn.Parameter(
                torch.ones(self.num_kernels)
            )

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        self.norm = nn.GroupNorm(
            8,
            out_ch
        )

        self.act = nn.SiLU()

    def forward(
        self,
        x,
        return_weights=False
    ):

        B, C, D, H, W = x.shape

        if not self.use_attention:

            # ----------------------------------------------------
            # Global learned routing
            # ----------------------------------------------------
    
            weights = F.softmax(
                self.alpha,
                dim=0
            )
    
            # ----------------------------------------------------
            # Sequential expert execution
            # ----------------------------------------------------
    
            y = self.conv_suite.forward_sequential(
                x,
                weights
            )
    
            # ----------------------------------------------------
            # Normalize + activation
            # ----------------------------------------------------
    
            y = self.norm(y)
            y = self.act(y)
    
            # ----------------------------------------------------
            # Return
            # ----------------------------------------------------
    
            if return_weights:
    
                weights_spatial = weights.view(
                    1,
                    self.num_kernels,
                    1,
                    1,
                    1
                ).expand(
                    B,
                    -1,
                    D,
                    H,
                    W
                )
    
                return y, weights_spatial
    
            return y
        
        # ====================================================
        # High-frequency extraction
        # ====================================================

        hf = (
            x
            - F.avg_pool3d(
                x,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

        hf = torch.clamp(
            hf,
            -3.0,
            3.0
        )

        # ====================================================
        # Normalize
        # ====================================================

        x = (
            x
            - x.mean(
                dim=(2, 3, 4),
                keepdim=True
            )
        ) / (
            x.std(
                dim=(2, 3, 4),
                keepdim=True
            ) + 1e-5
        )

        # ====================================================
        # Inject high frequency
        # ====================================================

        x = x + 0.5 * hf

        # ====================================================
        # Reduced spatial representation
        # ====================================================

        x_small = F.interpolate(
            x,
            scale_factor=(1, 0.5, 0.5),
            mode="trilinear",
            align_corners=False
        )

        x_small = (
            x_small
            - x_small.mean(
                dim=(2, 3, 4),
                keepdim=True
            )
        )

        # ====================================================
        # Channel reduction
        # ====================================================

        x_low = self.reduce(x_small)

        # ====================================================
        # High-frequency component in reduced space
        # ====================================================

        x_high = (
            x_low
            - F.avg_pool3d(
                x_low,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

        x_low = x_low / (
            x_low.std(
                dim=(2, 3, 4),
                keepdim=True
            ) + 1e-5
        )

        x_high = x_high / (
            x_high.std(
                dim=(2, 3, 4),
                keepdim=True
            ) + 1e-5
        )

        x_small = x_low + 0.5 * x_high

        _, _, D_s, H_s, W_s = x_small.shape

        # ====================================================
        # EXPERTS
        #
        # Kept unchanged intentionally.
        # ====================================================

        feats = self.conv_suite(x)

        
        # ====================================================
        # ROUTING
        # ====================================================

        if self.use_attention:

            (
                tokens,
                (Nd, Nh, Nw),
                (D_pad, H_pad, W_pad)
            ) = self.window_pool(x_small)

            # ------------------------------------------------
            # Attention
            # ------------------------------------------------

            logits = self.attn(tokens)

            # ------------------------------------------------
            # Restore window → spatial layout
            # ------------------------------------------------

            logits = logits.reshape(
                B,
                Nd,
                Nh,
                Nw,
                self.num_kernels
            )

            logits = logits.permute(
                0,
                4,
                1,
                2,
                3
            )

            # ------------------------------------------------
            # Restore padded spatial resolution
            # ------------------------------------------------

            logits = F.interpolate(
                logits,
                size=(
                    D_pad,
                    H_pad,
                    W_pad
                ),
                mode="trilinear",
                align_corners=False
            )

            # ------------------------------------------------
            # Crop
            # ------------------------------------------------

            logits = logits[
                :,
                :,
                :D_s,
                :H_s,
                :W_s
            ]

            # ------------------------------------------------
            # Upscale routing to feature resolution
            # ------------------------------------------------

            logits = F.interpolate(
                logits,
                size=(
                    D,
                    H,
                    W
                ),
                mode="trilinear",
                align_corners=False
            )

            # ------------------------------------------------
            # Feature strength
            # ------------------------------------------------

            feat_strength = x_small.abs().mean(
                dim=1,
                keepdim=True
            )

            feat_strength = F.interpolate(
                feat_strength,
                size=logits.shape[2:],
                mode="trilinear",
                align_corners=False
            )

            logits = (
                logits
                + 0.3 * feat_strength
            )

            # ------------------------------------------------
            # Normalize routing logits
            # ------------------------------------------------

            logits = (
                logits
                - logits.mean(
                    dim=(2, 3, 4),
                    keepdim=True
                )
            )

            logits = logits / (
                logits.std(
                    dim=1,
                    keepdim=True
                ) + 1e-5
            )

            # ------------------------------------------------
            # Stochastic symmetry breaking
            #
            # Kept because this changes training behavior.
            # ------------------------------------------------

            logits = (
                logits
                + 0.01 * torch.randn_like(logits)
            )

            weights = F.softmax(
                logits / 0.8,
                dim=1
            )

        else:

            # ------------------------------------------------
            # Global learned routing
            # ------------------------------------------------

            weights = F.softmax(
                self.alpha,
                dim=0
            )

            weights = weights.view(
                1,
                self.num_kernels,
                1,
                1,
                1
            )

            weights = weights.expand(
                B,
                -1,
                D,
                H,
                W
            )

        # ====================================================
        # Expert mixing
        #
        # Avoid an unnecessary zeros_like allocation.
        # ====================================================

        y = (
            weights[:, 0:1]
            * feats[0]
        )

        for i in range(
            1,
            self.num_kernels
        ):

            y = (
                y
                + weights[:, i:i+1]
                * feats[i]
            )

        # ====================================================
        # Output
        # ====================================================

        y = self.act(
            self.norm(y)
        )

        if return_weights:

            return y, weights

        return y


# ============================================================
# Spatial downsampling
# ============================================================

class SpatialDownsample3D(nn.Module):

    def __init__(self):

        super().__init__()

        self.pool = nn.AvgPool3d(
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2)
        )

    def forward(self, x):

        return self.pool(x)
