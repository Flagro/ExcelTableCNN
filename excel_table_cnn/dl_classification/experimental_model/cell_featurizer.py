from torch import nn


class CellFeaturizer(nn.Module):
    def __init__(self, in_channels=20, out_channels=32):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
