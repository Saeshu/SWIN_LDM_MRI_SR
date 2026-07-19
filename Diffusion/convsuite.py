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
        
        self.time_to_spatial = nn.Sequential(
            nn.Linear(time_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )
        ####################################################
        # Router
        ####################################################
        
        self.router = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(64, 3, kernel_size=1),
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
        te_spatial = self.time_to_spatial(te)

        # [B,64,1,1,1]
        te_spatial = te_spatial.unsqueeze(-1)\
                               .unsqueeze(-1)\
                               .unsqueeze(-1)
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
            # print(anatomy.shape)
            
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

           
        
            ################################################
            # Refine anatomy representation
            ################################################
            anatomy_before = anatomy.detach().clone()
            anatomy = anatomy + 0.1 * gate_map * h_feat
            
          
    
        
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

        gamma_map = gamma_scalar.view(-1,1,1,1,1)

        # Broadcast timestep automatically
        routing_feat = anatomy + gamma_map * te_spatial

        ####################################################
        # Routing
        ####################################################

        logits = self.router(routing_feat)
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(
                logits,
                size=x.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
        # Softmax over experts
        gates = torch.softmax(logits, dim=1)

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
        # print("gates new: ", gates.shape)
        ####################################################
        # Return raw gates if requested
        ####################################################
    
        if return_gates:
            # print(
            #     f"Gate mean : {gate.mean():.3f}",
            #     f"Gate std : {gate.std():.3f}",
            # ) 
            gate_out = gates.detach()
            # print(gate_out.shape)
        ####################################################
        # Experts
        ####################################################

        spatial = self.spatial(x)
        mid = self.mid(x)
        long = self.long(x)

        ####################################################
        # Broadcast gates
        ####################################################
        
        g0 = gates[:,0:1]
        g1 = gates[:,1:2]
        g2 = gates[:,2:3]
        # print("g0      :", g0.shape)
        # print("g1      :", g1.shape)
        # print("g2      :", g2.shape)
        
        # print("spatial :", spatial.shape)
        # print("mid     :", mid.shape)
        # print("long    :", long.shape)
        out = (
            g0 * spatial +
            g1 * mid +
            g2 * long
        )
        ####################################################
        # Mixture of experts
        ####################################################

        # out = (
        #     gates[:,0] * spatial
        #     +
        #     gates[:,1] * mid
        #     +
        #     gates[:,2] * long
        # )

        ####################################################
        # Return
        ####################################################

        if return_gates:
            # print(gate_out.mean(dim=0))
            return out, gate_out, routing_feat.detach()

        return out
