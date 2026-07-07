"""Fully convolutional backbone for spreadsheet tensors.

Stride 1 throughout — the paper's key adaptation: cell-level resolution is
preserved so boundaries can be located exactly. GroupNorm instead of
BatchNorm because detection training runs with batch size 1.

With ``use_grid_context=True`` (default) the backbone additionally exploits
grid structure — this project's own extension over TableSense (see
``grid_context.py``): derived row/column priors, a dilated middle stage
(receptive field ~11 → ~30+ cells at zero parameter cost; corpus median
table height is 21 rows), and an axial strip-pooling block that gives every
cell a summary of its entire row and column.
"""

import torch.nn as nn

from .grid_context import NUM_DERIVED_CHANNELS, AxialStripBlock, DerivedChannels


class FCNBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 256,
        norm_groups: int = 8,
        use_grid_context: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels  # data channels, before derived priors
        self.use_grid_context = use_grid_context

        if use_grid_context:
            self.derived = DerivedChannels()
            conv_in = in_channels + NUM_DERIVED_CHANNELS
            dilations = (1, 2, 4)
        else:
            self.derived = None
            conv_in = in_channels
            dilations = (1, 1, 1)

        layers = [
            nn.Conv2d(conv_in, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        ]
        for dilation in dilations:
            layers += [
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3,
                          padding=dilation, dilation=dilation),
                nn.GroupNorm(norm_groups, hidden_channels),
                nn.ReLU(inplace=True),
            ]
        if use_grid_context:
            layers.append(AxialStripBlock(hidden_channels))
        layers.append(nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1))
        self.body = nn.Sequential(*layers)
        self.out_channels = out_channels  # required by torchvision's FasterRCNN

    def forward(self, x):
        if self.derived is not None:
            x = self.derived(x)
        return self.body(x)
