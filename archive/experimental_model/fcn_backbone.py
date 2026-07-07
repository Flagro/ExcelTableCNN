import torch.nn as nn
import torch.nn.functional as F


class FCNBackbone(nn.Module):
    def __init__(self, in_channels=32, out_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x
