class EpsilonUNet3D(nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        self.conv1 = nn.Conv3d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv3d(ch, ch, 3, padding=1)
        self.conv3 = nn.Conv3d(ch, ch, 3, padding=1)

    def forward(self, x, t):
        # t is unused for now (we add it later)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.conv3(x)
