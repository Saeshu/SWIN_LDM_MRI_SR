import torch
import torch.nn as nn

# --------------------------------------------------
# Residual Block
# --------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch, ch, 3, padding=1),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


# --------------------------------------------------
# Latent Perceptual Network
# --------------------------------------------------

class LatentPerceptualNet(nn.Module):

    def __init__(self, in_ch=2):

        super().__init__()

        # ---------------- Encoder ---------------- #

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            ResBlock(32),
        )

        self.down1 = nn.Conv3d(
            32,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.conv2 = nn.Sequential(
            nn.ReLU(inplace=True),
            ResBlock(64),
        )

        self.down2 = nn.Conv3d(
            64,
            128,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.conv3 = nn.Sequential(
            nn.ReLU(inplace=True),
            ResBlock(128),
        )

        # ---------------- Decoder ---------------- #

        self.up1 = nn.ConvTranspose3d(
            128,
            64,
            kernel_size=2,
            stride=2,
        )

        self.dec1 = nn.Sequential(
            ResBlock(64),
        )

        self.up2 = nn.ConvTranspose3d(
            64,
            32,
            kernel_size=2,
            stride=2,
        )

        self.dec2 = nn.Sequential(
            ResBlock(32),
        )

        self.out = nn.Conv3d(
            32,
            in_ch,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):

        # ---------- Encoder ---------- #

        f1 = self.conv1(x)

        x = self.down1(f1)

        f2 = self.conv2(x)

        x = self.down2(f2)

        f3 = self.conv3(x)

        # ---------- Decoder ---------- #

        x = self.up1(f3)

        x = self.dec1(x)

        x = self.up2(x)

        x = self.dec2(x)

        recon = self.out(x)

        return [f1, f2, f3], recon
