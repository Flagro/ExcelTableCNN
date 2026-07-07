"""Fully convolutional backbone for spreadsheet tensors.

Stride 1 throughout — the paper's key adaptation: cell-level resolution is
preserved so boundaries can be located exactly. GroupNorm instead of
BatchNorm because detection training runs with batch size 1.
"""

import torch.nn as nn


class FCNBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 256,
        num_blocks: int = 3,
        norm_groups: int = 8,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_blocks):
            layers += [
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                nn.GroupNorm(norm_groups, hidden_channels),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1))
        self.body = nn.Sequential(*layers)
        self.out_channels = out_channels  # required by torchvision's FasterRCNN

    def forward(self, x):
        return self.body(x)
