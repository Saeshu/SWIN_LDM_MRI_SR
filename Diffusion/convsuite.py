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
        # Encode w_E2
        ####################################################

        self.we2_pool = nn.Sequential(

            nn.Conv3d(
                we2_channels,
                16,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        
            nn.SiLU(),
        
            nn.Conv3d(
                16,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        
            nn.SiLU(),
        
            nn.AdaptiveAvgPool3d((2,2,2)),
        
            nn.Flatten(),
        )

        ####################################################
        # Project anatomy embedding
        ####################################################

        self.we2_proj = nn.Sequential(

            nn.Linear(
                32 * 2 * 2 * 2,
                time_dim,
            ),
        
            nn.SiLU(),
        
            nn.Linear(
                time_dim,
                time_dim,
            ),
        )

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

        ####################################################
        # Anatomy embedding
        ####################################################

        if w_e2 is None:
            print("we2 none")
            we2_feat = torch.zeros_like(te)

        else:

            we2_feat = self.we2_pool(w_e2)

        we2_feat = self.we2_proj(
            we2_feat,
        )

        te = F.normalize(te, dim=-1)
        we2_feat = F.normalize(we2_feat, dim=-1)


        ####################################################
        # SNR weighting
        ####################################################
        # print(
        #     te.norm(dim=1).mean()
        # )
        
        # print(
        #     we2_feat.norm(dim=1).mean()
        # )

        
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
        # print(
        #     gamma_scalar.mean()
        # )
        # we2_feat = gamma_scalar * we2_feat

        ####################################################
        # Learn fusion
        ####################################################

        fusion = torch.cat(
            [
                te,
                we2_feat,
            ],
            dim=-1,
        )

        gate = torch.sigmoid(
            self.gate_fc(fusion)
        )

        # print("gate-te", (gate * te).norm(dim=1).mean())
        
        # print("1-gate te", ((1-gate) * gamma_scalar * we2_feat).norm(dim=1).mean())
        routing_feat = (
            gate * te
            +
            (1.0 - gate) * we2_feat * gamma_scalar
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
            # print(
            #     f"Gate mean : {gate.mean():.3f}",
            #     f"Gate std : {gate.std():.3f}",
            # ) 
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
