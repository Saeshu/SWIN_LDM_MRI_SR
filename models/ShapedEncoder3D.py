import torch
import torch.nn as nn
import torch.nn.functional as F


class AnisotropicConvSuite(nn.Module):
    def __init__(self, in_ch, out_ch, depth_kernels=(3, 5, 7)):
        super().__init__()
        
        self.conv_3x3x1 = nn.Conv3d(
            in_ch, out_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1)
        )

        self.depth_convs = nn.ModuleList([
            nn.Conv3d(
                in_ch, out_ch,
                kernel_size=(k, 1, 1),
                padding=(k // 2, 0, 0)
            )
            for k in depth_kernels
        ])

        self.conv_1x1x1 = nn.Conv3d(in_ch, out_ch, kernel_size=1)

        self.num_paths = 1 + len(depth_kernels) + 1

    def forward(self, x):
        feats = []

        feats.append(self.conv_3x3x1(x))

        for conv in self.depth_convs:
            feats.append(conv(x))

        feats.append(self.conv_1x1x1(x))

        return feats


class WindowPool3D(nn.Module):
    def __init__(self, window_size=(1, 11, 11)):
        super().__init__()
        self.window_size = window_size
        
    def forward(self, x):
        B, C, D, H, W = x.shape
        wd, wh, ww = self.window_size
    
        # 🔥 clamp properly
        wd = min(wd, D)
        wh = min(wh, H)
        ww = min(ww, W)
    
        # 🔥 IMPORTANT: store for reuse
        self._last_window = (wd, wh, ww)
    
        x = x.unfold(2, wd, wd) \
             .unfold(3, wh, wh) \
             .unfold(4, ww, ww)
    
        x = x.contiguous().view(B, C, -1, wd * wh * ww)
    
        tokens = x.mean(dim=-1) + 0.5 * x.std(dim=-1)
        tokens = tokens.permute(0, 2, 1)
    
        return tokens


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
        depth_kernels=(3, 5, 7),
        window_size=(1, 21, 21),
        use_attention=True
    ):
        super().__init__()

        self.conv_suite = AnisotropicConvSuite(
            in_ch, out_ch, depth_kernels
        )
        reduced_ch = max(1, in_ch // 2)
        self.reduce = nn.Conv3d(in_ch, reduced_ch, 1)

        self.use_attention = use_attention
        self.num_kernels = self.conv_suite.num_paths
        self.window_size = window_size
        
        if use_attention:
            self.window_pool = WindowPool3D(window_size)
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
    
        x_small = F.avg_pool3d(x, kernel_size=(2,4,4), stride=(2,4,4))
        x_small = self.reduce(x_small)
    
        B, C_s, D_s, H_s, W_s = x_small.shape
    
        feats = self.conv_suite(x)
    
        if self.use_attention:
    
            # -----------------------------
            # Tokenization
            # -----------------------------
            tokens = self.window_pool(x_small)  # [B, N, C]
    
            # positional bias
            pos = torch.linspace(-1, 1, tokens.shape[1], device=tokens.device)
            pos = pos.unsqueeze(0).unsqueeze(-1)
            tokens = tokens + 0.3 * pos
    
            # -----------------------------
            # Attention → logits
            # -----------------------------
            logits = self.attn(tokens)  # [B, N, K]
    
            # -----------------------------
            # 🔥 Spatial residual (CRITICAL)
            # -----------------------------
            w_local = x_small.mean(dim=1, keepdim=True)  # [B,1,D_s,H_s,W_s]
    
            w_local = w_local.view(B, 1, -1).permute(0, 2, 1)  # [B, N, 1]
            w_local = w_local.expand(-1, -1, logits.shape[-1])  # [B, N, K]
    
            logits = logits + 0.1 * w_local   # 🔥 THIS LINE WAS MISSING
    
            # -----------------------------
            # Reshape → spatial map
            # -----------------------------
            wd, wh, ww = self.window_pool._last_window
    
            Nd = D_s // wd
            Nh = H_s // wh
            Nw = W_s // ww
    
            assert tokens.shape[1] == Nd * Nh * Nw
    
            logits = logits.reshape(B, Nd, Nh, Nw, self.num_kernels)
            logits = logits.permute(0, 4, 1, 2, 3)  # [B, K, Nd, Nh, Nw]
    
            logits = F.interpolate(
                logits,
                size=(D, H, W),
                mode="trilinear",
                align_corners=False
            )
    
            weights = logits   # 🔥 IMPORTANT: still logits (no softmax)
    
        else:
            weights = F.softmax(self.alpha, dim=0)
            weights = weights.view(1, self.num_kernels, 1, 1, 1)
            weights = weights.expand(B, -1, D, H, W)
    
        # -----------------------------
        # Mixing
        # -----------------------------
        y = sum(weights[:, i:i+1] * f for i, f in enumerate(feats))
    
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
