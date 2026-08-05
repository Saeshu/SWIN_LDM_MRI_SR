import torch
import torch.nn as nn
import torch.nn.functional as F


class AnisotropicConvSuite(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.kernels = nn.ModuleList([

            # 1. low-pass (structure)
            nn.Sequential(
                nn.AvgPool3d((1,3,3), stride=1, padding=(0,1,1)),
                nn.Conv3d(in_ch, out_ch, 1)   # 🔥 FIX
            ),
        
            # 2. high-pass (edges)
            nn.Conv3d(in_ch, out_ch, 1),
        
            # 3. spatial
            nn.Conv3d(in_ch, out_ch, (1,3,3), padding=(0,1,1)),
        
            # 4. depth
            nn.Conv3d(in_ch, out_ch, (3,1,1), padding=(1,0,0)),
        
            # 5. identity
            nn.Conv3d(in_ch, out_ch, 1)
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
    def __init__(self, window_size=None, shift=False):
        super().__init__()
        self.window_size = window_size
        self.shift = shift

    def forward(self, x):
        B, C, D, H, W = x.shape
        # wd, wh, ww = self.window_size
        if self.window_size is None:
            target_grid = (32, 4, 4)   # (D, H, W)

            D, H, W = x.shape[2:]
            
            wd = max(1, D // target_grid[0])
            wh = max(3, H // target_grid[1])
            ww = max(3, W // target_grid[2])
            
            # keep odd sizes for H/W
            if wh % 2 == 0:
                wh += 1
            if ww % 2 == 0:
                ww += 1
            
            # depth can stay any integer
            wd = min(wd, D)
            wh = min(wh, H)
            ww = min(ww, W)
            # print("window size: ", wd, ",", wh, ",", ww)         
        else:
            wd, wh, ww = self.window_size
        # -----------------------------
        # Compute padding
        # -----------------------------
        pad_d = (wd - D % wd) % wd
        pad_h = (wh - H % wh) % wh
        pad_w = (ww - W % ww) % ww

        x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))

        D_pad, H_pad, W_pad = x.shape[2:]

        # -----------------------------
        # Unfold into windows
        # -----------------------------
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

        # -----------------------------
        # Tokenization
        # -----------------------------
        mean = x.mean(dim=-1)
        std = x.std(dim=-1)

        raw_tokens = mean + 0.1 * std

        tokens = raw_tokens / (
            raw_tokens.std(dim=-1, keepdim=True) + 1e-6
        )

        tokens = tokens.permute(0, 2, 1)

        return (
            tokens,
            (Nd, Nh, Nw),
            (wd, wh, ww),
            (D_pad, H_pad, W_pad),
            mean,
            std,
            raw_tokens,
        )
    
    
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
        window_size=None,
        use_attention=True,
        shift=False
    ):
        super().__init__()
        if use_attention:
            self.window_pool = WindowPool3D(window_size, shift=False)

        self.conv_suite = AnisotropicConvSuite(
            in_ch, out_ch
        )
        reduced_ch = max(1, in_ch // 2)
        self.reduce = nn.Conv3d(in_ch, reduced_ch, 1)

        self.use_attention = use_attention
        self.num_kernels = self.conv_suite.num_paths
        self.window_size = window_size
        
        if use_attention:
            self.window_pool = WindowPool3D(window_size, shift=False)
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
        # print("window size:", self.window_size)
        # -----------------------------
        # HF from raw
        # -----------------------------
        hf = x - F.avg_pool3d(x, 3, 1, 1)
        hf = torch.clamp(hf, -3.0, 3.0)
    
        # -----------------------------
        # Normalize
        # -----------------------------
        x = (x - x.mean(dim=(2,3,4), keepdim=True)) / (
            x.std(dim=(2,3,4), keepdim=True) + 1e-5
        )
    
        # -----------------------------
        # Inject HF
        # -----------------------------
        x = x + 0.5 * hf
    
        # -----------------------------
        # Downsample
        # -----------------------------
        x_small = F.interpolate(
            x,
            scale_factor=(1, 0.5, 0.5),
            mode="trilinear",
            align_corners=False
        )
        x_small = x_small - x_small.mean(dim=(2,3,4), keepdim=True)
            
        # -----------------------------
        # Reduce
        # -----------------------------
        x_low = self.reduce(x_small)
    
        # -----------------------------
        # HF in SAME space
        # -----------------------------
        x_high = x_low - F.avg_pool3d(x_low, 3, 1, 1)
        x_low  = x_low  / (x_low.std(dim=(2,3,4), keepdim=True) + 1e-5)
        x_high = x_high / (x_high.std(dim=(2,3,4), keepdim=True) + 1e-5) 
        x_small = x_low + 0.5 * x_high
        # print("HF:", hf.std().item())
        # print("HF std before reduce:", x_high.std().item())
        # print("After reduce:", x_small.std().item())
        # print("x_small device:", x_small.device)
        # print("reduce device:", next(self.reduce.parameters()).device)
        B, C_s, D_s, H_s, W_s = x_small.shape
        # print("x_small: ", x_small.shape)
        feats = self.conv_suite(x)
    
        if self.use_attention:
    
            (
                  tokens,
                  (Nd, Nh, Nw),
                  (wd, wh, ww),
                  (D_pad, H_pad, W_pad),
                  mean,
                  std,
                  raw_tokens,
              ) = self.window_pool(x_small)

            D_orig, H_orig, W_orig = x_small.shape[2:]

            if self.training:
                perm = torch.randperm(tokens.shape[1], device=tokens.device)
                tokens = tokens[:, perm]

            # positional bias
            coords = torch.stack(torch.meshgrid(
            torch.linspace(-1,1,Nd, device=tokens.device),
            torch.linspace(-1,1,Nh, device=tokens.device),
            torch.linspace(-1,1,Nw, device=tokens.device),
            indexing='ij'
        ), dim=-1)
        
            coords = coords.view(-1, 3)
            
            if coords.shape[-1] < tokens.shape[-1]:
                coords = F.pad(coords, (0, tokens.shape[-1] - 3))
            
            # tokens = tokens + 0.1 * coords.unsqueeze(0)
            
            
            # -----------------------------
            # Attention → logits
            # -----------------------------
            logits = self.attn(tokens) 
            # print("mean logits: ", logits.mean())
            # print("std logits: ", logits.std()) # [B, N, K]
            raw_logits = logits.detach().clone()
            if self.training:
                inv_perm = torch.argsort(perm)
                logits = logits[:, inv_perm]
            # -----------------------------
            # Spatial residual (reuse tokens)
            # -----------------------------
            # w_local_tokens = tokens.mean(dim=-1, keepdim=True)  # [B, N, 1]
            # w_local_tokens = w_local_tokens.expand(-1, -1, logits.shape[-1])  # [B, N, K]
            
            # logits = logits + 0.3 * w_local_tokens
            
            # -----------------------------
            # Reshape → spatial map
            # -----------------------------
            assert tokens.shape[1] == Nd * Nh * Nw
            
            logits = logits.reshape(B, Nd, Nh, Nw, self.num_kernels)
            logits = logits.permute(0, 4, 1, 2, 3)  # [B, K, Nd, Nh, Nw]
            
            logits = F.interpolate(
                logits,
                size=(D_pad, H_pad, W_pad),
                mode="trilinear",
                align_corners=False
            )
            
            # step 2: crop to original x_small
            logits = logits[:, :, :D_s, :H_s, :W_s]
            
            # 🔥 step 3: upscale to FULL resolution (CRITICAL)
            logits = F.interpolate(
                logits,
                size=(D, H, W),
                mode="trilinear",
                align_corners=False
            )
            
            feat_strength = x_small.abs().mean(dim=1, keepdim=True)
            
            feat_strength = F.interpolate(
                feat_strength,
                size=logits.shape[2:],
                mode="trilinear",
                align_corners=False
            )
            
            logits = logits + 0.3 * feat_strength
            
            # normalize (your style)
            # remove global spatial bias
            logits = logits - logits.mean(dim=(2,3,4), keepdim=True)
            logits = logits / (logits.std(dim=1, keepdim=True) + 1e-5)
            logits = logits + 0.01 * torch.randn_like(logits)
            weights = F.softmax(logits / 0.8, dim=1)
            # weights = torch.ones(
            #     B,
            #     self.num_kernels,
            #     D,
            #     H,
            #     W,
            #     device=x.device
            # )
            
            # weights /= self.num_kernels
    
        else:
            weights = F.softmax(self.alpha, dim=0)
            weights = weights.view(1, self.num_kernels, 1, 1, 1)
            weights = weights.expand(B, -1, D, H, W)
    
        # -----------------------------
        # Mixing
        # -----------------------------
       # just aggregate features normally (no kernel mixing here)
        # weights = F.softmax(weights, dim=1)
        assert weights.shape[1] == len(feats)
        y = torch.zeros_like(feats[0])
        for i, f in enumerate(feats):
            y = y + weights[:, i:i+1] * f
        y = self.act(self.norm(y))
        
        if return_weights:
            return (
                  y,
                  weights
              )
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
