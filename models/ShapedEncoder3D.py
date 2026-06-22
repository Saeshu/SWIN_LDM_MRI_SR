import torch
import torch.nn as nn
import torch.nn.functional as F


class AnisotropicConvSuite(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.kernels = nn.ModuleList([
            # full spatial
            nn.Conv3d(in_ch, out_ch, kernel_size=(1,3,3), padding=(0,1,1)),

            # directional spatial
            nn.Conv3d(in_ch, out_ch, kernel_size=(1,3,1), padding=(0,1,0)),  # vertical
            nn.Conv3d(in_ch, out_ch, kernel_size=(1,1,3), padding=(0,0,1)),  # horizontal

            # depth
            nn.Conv3d(in_ch, out_ch, kernel_size=(3,1,1), padding=(1,0,0)),

            # channel mixer
            nn.Conv3d(in_ch, out_ch, kernel_size=1)
        ])

        self.num_paths = len(self.kernels)

    def forward(self, x):
        return [conv(x) for conv in self.kernels]

def shifted_pad(x, shift_d, shift_h, shift_w):
    """
    Non-circular shift using padding (no wrap-around).
    """
    B, C, D, H, W = x.shape

    # pad on the "front" side
    x = F.pad(x, (shift_w, 0, shift_h, 0, shift_d, 0))

    # crop back to original size
    x = x[:, :, :D, :H, :W]

    return x
    
class WindowPool3D(nn.Module):
    def __init__(self, window_size=(1, 11, 11), shift=False):
        super().__init__()
        self.window_size = window_size
        self.shift = shift
        
    def forward(self, x):
        B, C, D, H, W = x.shape
        wd, wh, ww = self.window_size

        # 🔥 SHIFT (correct)
        if self.shift:
            shift_d = wd // 2
            shift_h = wh // 2
            shift_w = ww // 2
            x = shifted_pad(x, shift_d, shift_h, shift_w)
        # Clamp window size
        wd = min(wd, D)
        wh = min(wh, H)
        ww = min(ww, W)

        # Ensure divisibility (IMPORTANT)
        D_trim = (D // wd) * wd
        H_trim = (H // wh) * wh
        W_trim = (W // ww) * ww

        x = x[:, :, :D_trim, :H_trim, :W_trim]

        # Unfold into windows
        x = x.unfold(2, wd, wd) \
             .unfold(3, wh, wh) \
             .unfold(4, ww, ww)

        # [B, C, Nd, Nh, Nw, wd, wh, ww]
        Nd, Nh, Nw = x.shape[2:5]

        x = x.contiguous().view(B, C, Nd * Nh * Nw, wd * wh * ww)

        # 🔥 Stable tokenization
        mean = x.mean(dim=-1)
        std = x.std(dim=-1)

        tokens = mean + 0.3 * std   # slightly safer weight

        # Normalize tokens (VERY IMPORTANT for attention)
        tokens = tokens / (tokens.std(dim=-1, keepdim=True) + 1e-6)

        # [B, N, C]
        tokens = tokens.permute(0, 2, 1)

        # 🔥 return shape info explicitly
        return tokens, (Nd, Nh, Nw), (wd, wh, ww)


class KernelMixingAttention(nn.Module):
    def __init__(self, embed_dim, num_kernels, num_heads=4):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.proj = nn.Linear(embed_dim, num_kernels)
        
        #print("attn embed:", self.attn.embed_dim)
        #print("attn num_heads:", self.attn.num_heads)
    def forward(self, tokens):
        attn_out, _ = self.attn(tokens, tokens, tokens)
        logits = self.proj(attn_out)
        # weights = F.softmax(logits, dim=-1)
        
        return logits  # [B, N, K]

   
class AnisotropicSwinBlock(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        window_size=(1, 11, 11),
        use_attention=True
        shift=False
    ):
        super().__init__()
        if use_attention:
            self.window_pool = WindowPool3D(window_size, shift=True)

        self.conv_suite = AnisotropicConvSuite(
            in_ch, out_ch
        )
        reduced_ch = max(1, in_ch // 2)
        self.reduce = nn.Conv3d(in_ch, reduced_ch, 1)

        self.use_attention = use_attention
        self.num_kernels = self.conv_suite.num_paths
        self.window_size = window_size
        
        if use_attention:
            self.window_pool = WindowPool3D(window_size, shift=True)
            self.attn = KernelMixingAttention(
                embed_dim=reduced_ch,
                num_kernels=self.num_kernels
            )
        else:
            self.alpha = nn.Parameter(torch.ones(self.num_kernels))

        self.norm = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()

    def forward(self, x, return_weights=False):
        B, C, D, H, W = x.shape
        # print("encoder K:", w_E2.shape[1])

        x_small = F.avg_pool3d(x, kernel_size=(2,4,4), stride=(2,4,4))
        x_small = self.reduce(x_small)
    
        B, C_s, D_s, H_s, W_s = x_small.shape
    
        feats = self.conv_suite(x)
    
        if self.use_attention:
    
            tokens, (Nd, Nh, Nw), (wd, wh, ww) = self.window_pool(x_small)

            # positional bias
            pos = torch.linspace(-1, 1, tokens.shape[1], device=tokens.device)
            pos = pos.unsqueeze(0).unsqueeze(-1)
            tokens = tokens + 0.2 * pos
            
            # -----------------------------
            # Attention → logits
            # -----------------------------
            logits = self.attn(tokens)  # [B, N, K]
            
            # -----------------------------
            # Spatial residual (reuse tokens)
            # -----------------------------
            w_local_tokens = tokens.mean(dim=-1, keepdim=True)  # [B, N, 1]
            w_local_tokens = w_local_tokens.expand(-1, -1, logits.shape[-1])  # [B, N, K]
            
            logits = logits + 0.3 * w_local_tokens
            
            # -----------------------------
            # Reshape → spatial map
            # -----------------------------
            assert tokens.shape[1] == Nd * Nh * Nw
            
            logits = logits.reshape(B, Nd, Nh, Nw, self.num_kernels)
            logits = logits.permute(0, 4, 1, 2, 3)  # [B, K, Nd, Nh, Nw]
            
            logits = F.interpolate(
                logits,
                size=(D, H, W),
                mode="trilinear",
                align_corners=False
            )
            
            # normalize (your style)
            weights = logits / (logits.std(dim=1, keepdim=True) + 1e-5)
    
        else:
            weights = F.softmax(self.alpha, dim=0)
            weights = weights.view(1, self.num_kernels, 1, 1, 1)
            weights = weights.expand(B, -1, D, H, W)
    
        # -----------------------------
        # Mixing
        # -----------------------------
       # just aggregate features normally (no kernel mixing here)
        weights = F.softmax(weights, dim=1)
        assert weights.shape[1] == len(feats)
        y = torch.zeros_like(feats[0])
        for i, f in enumerate(feats):
            y = y + weights[:, i:i+1] * f
        y = self.act(self.norm(y))
    
        if return_weights:
            return y, weights
    
        return y
    
            

class SpatialDownsample3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool3d(
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2)
        )

    def forward(self, x):
        return self.pool(x)
