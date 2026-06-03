class LatentPerceptualNet(nn.Module):
    def __init__(self, in_ch=2):   # 🔥 change to 2
        super().__init__()

        self.enc = nn.Sequential(
            nn.Conv3d(in_ch, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 16, 3, padding=1),
            nn.ReLU(),
        )

        self.dec = nn.Sequential(
            nn.Conv3d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, in_ch, 3, padding=1),  # 🔥 match input channels
        )

    def forward(self, x):
        f = self.enc(x)
        recon = self.dec(f)
        return f, recon
