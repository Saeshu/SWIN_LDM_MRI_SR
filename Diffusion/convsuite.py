import torch
import torch.nn as nn
import torch.nn.functional as F

from Diffusion.schedule import SinusoidalTimeEmbedding


############################################################
# Experts
############################################################

class SpatialSuite(nn.Module):
    def __init__(self, c):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv3d(c, c, (3,3,1), padding=(1,1,0)),
            nn.SiLU(),
            nn.Conv3d(c, c, (1,3,1), padding=(0,1,0)),
            nn.SiLU(),
            nn.Conv3d(c, c, (3,1,1), padding=(1,0,0)),
        )

    def forward(self, x):
        return self.net(x)


class MidSliceSuite(nn.Module):
    def __init__(self, c):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv3d(c, c, (1,1,3), padding=(0,0,1)),
            nn.SiLU(),
            nn.Conv3d(c, c, (1,1,5), padding=(0,0,2)),
        )

    def forward(self, x):
        return self.net(x)


class LongSliceSuite(nn.Module):
    def __init__(self, c):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv3d(c, c, (1,1,7), padding=(0,0,3)),
            nn.SiLU(),
            nn.Conv3d(c, c, (1,1,9), padding=(0,0,4)),
        )

    def forward(self, x):
        return self.net(x)


class SpatialCrossAttention(nn.Module):

    def __init__(self, channels=64):

        super().__init__()

        self.query = nn.Conv3d(
            channels,
            channels,
            1,
        )

        self.key = nn.Conv3d(
            channels,
            channels,
            1,
        )

        self.value = nn.Conv3d(
            channels,
            channels,
            1,
        )

        self.scale = channels ** -0.5

    def forward(
        self,
        anatomy,
        h,
    ):

        B,C,D,H,W = anatomy.shape

        q = self.query(anatomy).flatten(2)

        k = self.key(h).flatten(2)

        v = self.value(h).flatten(2)

        attn = torch.softmax(

            torch.bmm(
                q.transpose(1,2),
                k,
            ) * self.scale,

            dim=-1,
        )

        out = torch.bmm(
            attn,
            v.transpose(1,2),
        )

        out = out.transpose(1,2)

        out = out.view(
            B,
            C,
            D,
            H,
            W,
        )

        return anatomy + out

############################################################
# Time + Anatomy Aware Routing
############################################################

class TimeGatedConvSuite(nn.Module):

    def __init__(
        self,
        channels,
        time_dim=128,
        we2_channels=5,
    ):
        super().__init__()

        ####################################################
        # Time embedding
        ####################################################
        
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        
        ####################################################
        # Anatomy encoder
        ####################################################
        
        self.we2_encoder = nn.Sequential(
        
            nn.Conv3d(
                we2_channels,
                32,
                kernel_size=3,
                padding=1,
            ),
        
            nn.SiLU(),
        
            nn.Conv3d(
                32,
                64,
                kernel_size=3,
                padding=1,
            ),
        
            nn.SiLU(),
        )
        
        ####################################################
        # Project UNet bottleneck
        ####################################################
        
        self.h_proj = nn.Conv3d(
            channels,
            64,
            kernel_size=1,
        )
        
        ####################################################
        # Posterior-aware gating
        ####################################################
        
        self.posterior_gate = nn.Sequential(
        
            nn.Conv3d(
                64 + 64,
                64,
                kernel_size=3,
                padding=1,
            ),
        
            nn.SiLU(),
        
            nn.Conv3d(
                64,
                64,
                kernel_size=1,
            ),
        )
        
        ####################################################
        # Pool anatomy
        ####################################################
        
        self.we2_pool = nn.Sequential(
        
            nn.AdaptiveAvgPool3d((2,2,2)),
        
            nn.Flatten(),
        )
        
        ####################################################
        # Projection
        ####################################################
        
        self.we2_proj = nn.Sequential(
        
            nn.Linear(
                64 * 2 * 2 * 2,
                time_dim,
            ),
        
            nn.SiLU(),
        
            nn.Linear(
                time_dim,
                time_dim,
            ),
        )
        self.posterior_alpha = nn.Parameter(torch.tensor(0.0))
        ####################################################
        # Learn fusion gate
        ####################################################
        
        self.gate_fc = nn.Sequential(
        
            nn.Linear(
                2 * time_dim,
                time_dim,
            ),
        
            nn.SiLU(),
        
            nn.Linear(
                time_dim,
                time_dim,
            ),
        )
        
        ####################################################
        # Router
        ####################################################
        
        self.router = nn.Sequential(
        
            nn.Linear(
                time_dim,
                time_dim,
            ),
        
            nn.SiLU(),
        
            nn.Linear(
                time_dim,
                3,
            ),
        )

        ####################################################
        # Experts
        ####################################################

        self.spatial = SpatialSuite(channels)
        self.mid = MidSliceSuite(channels)
        self.long = LongSliceSuite(channels)

    ########################################################

    def forward(
        self,
        x,
        t,
        w_e2=None,
        gamma=None,
        expert_mask=None,
        return_gates=False,
    ):

        ####################################################
        # Time embedding
        ####################################################
        
        te = self.time_embed(t)
        
        te = F.normalize(
            te,
            dim=-1,
        )
        
        ####################################################
        # Anatomy branch
        ####################################################
        
        if w_e2 is None:
        
            we2_feat = torch.zeros_like(te)
        
        else:
        
            ################################################
            # Spatial anatomy features
            ################################################
        
            anatomy = self.we2_encoder(
                w_e2,
            )
        
            ################################################
            # Current UNet belief
            ################################################
        
            h_feat = self.h_proj(
                x,
            )
            h_feat = F.interpolate(
                h_feat,
                size=anatomy.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
            anatomy = F.normalize(anatomy, dim=1)

            h_feat = F.normalize(h_feat, dim=1)
            ################################################
            # Posterior-aware gating
            ################################################
            # print("anatomy :", anatomy.shape)
            # print("h_feat  :", h_feat.shape)
            print("Entered posterior gate")
            gate_map = torch.sigmoid(
            
                self.posterior_gate(
        
                    torch.cat(
                        [
                            anatomy,
                            h_feat,
                        ],
                        dim=1,
                    )
        
                )
        
            )

            print(
                "Posterior gate:",
                gate_map.mean().item(),
                gate_map.std().item(),
            )
        
            ################################################
            # Refine anatomy representation
            ################################################
            anatomy_before = anatomy.detach().clone()
            anatomy = anatomy + 0.1 * gate_map * h_feat
            delta = (
                anatomy - anatomy_before
            ).norm() / (
                anatomy_before.norm() + 1e-8
            )
            print(
                "Gate contribution:",
                (gate_map * h_feat).abs().mean().item()
            )
            print(
                "Relative anatomy update:",
                delta.item(),
            )
            print(
                "Update/max:",
                delta.abs().max().item(),
            )
            cos = F.cosine_similarity(

                anatomy_before.flatten(1),
            
                anatomy.flatten(1),
            
                dim=1,
            
            ).mean()
            
            print(
                "Cosine:",
                cos.item(),
            )

            print(
                "Anatomy norm:",
                anatomy.flatten(1).norm(dim=1).mean().item(),
            )

            print(
                "UNet norm:",
                 h_feat.flatten(1).norm(dim=1).mean().item(),
            )

            
            
            
            
            ################################################
            # Global descriptor
            ################################################
        
            we2_feat = self.we2_pool(
                anatomy,
            )
        
            we2_feat = self.we2_proj(
                we2_feat,
            )
        
            we2_feat = F.normalize(
                we2_feat,
                dim=-1,
            )
        
        ####################################################
        # Gamma
        ####################################################
        
        if gamma is None:
        
            gamma_scalar = torch.ones(
                te.shape[0],
                1,
                device=te.device,
                dtype=te.dtype,
            )
        
        else:
        
            gamma_scalar = gamma.reshape(
                gamma.shape[0],
                -1,
            ).mean(
                dim=1,
                keepdim=True,
            )
        
        ####################################################
        # Feature fusion
        ####################################################
        
        fusion = torch.cat(
            [
                te,
                we2_feat,
            ],
            dim=-1,
        )
        
        gate = torch.sigmoid(
            self.gate_fc(
                fusion,
            )
        )

        corr = F.cosine_similarity(

            anatomy.flatten(1),
        
            h_feat.flatten(1),
        
            dim=1,
        
        ).mean()
        
        print(
            "Anatomy-H correlation:",
            corr.item(),
        )
        
        routing_feat = (
        
            gate * te
        
            +
        
            gamma_scalar * (1.0 - gate) * we2_feat
        
        ) 

        

        ####################################################
        # Routing
        ####################################################

        logits = self.router(
            routing_feat,
        )

        gates = torch.softmax(
            logits,
            dim=-1,
        )

        ####################################################
        # Optional masking
        ####################################################

        if expert_mask is not None:

            if not torch.is_tensor(expert_mask):

                expert_mask = torch.tensor(
                    expert_mask,
                    device=gates.device,
                    dtype=gates.dtype,
                )

            gates = gates * expert_mask

            gates = gates / (
                gates.sum(
                    dim=-1,
                    keepdim=True,
                ) + 1e-8
            )

        ####################################################
        # Return raw gates if requested
        ####################################################
    
        if return_gates:
            print(
                f"Gate mean : {gate.mean():.3f}",
                f"Gate std : {gate.std():.3f}",
            ) 
            gate_out = gates.detach()

        ####################################################
        # Experts
        ####################################################

        spatial = self.spatial(x)
        mid = self.mid(x)
        long = self.long(x)

        ####################################################
        # Broadcast gates
        ####################################################

        gates = gates.unsqueeze(-1)\
                     .unsqueeze(-1)\
                     .unsqueeze(-1)\
                     .unsqueeze(-1)

        ####################################################
        # Mixture of experts
        ####################################################

        out = (
            gates[:,0] * spatial
            +
            gates[:,1] * mid
            +
            gates[:,2] * long
        )

        ####################################################
        # Return
        ####################################################

        if return_gates:
            # print(gate_out.mean(dim=0))
            return out, gate_out, routing_feat.detach()

        return out
