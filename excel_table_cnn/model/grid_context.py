"""Grid-context components

Spreadsheets are not natural images: table boundaries are *global
row/column events* (a separator is a whole blank column; a header restyles a
whole row). Two cheap components exploit that grid structure directly:

- ``DerivedChannels`` prepends four prior channels computed from the input
  tensor itself: row fill-density, column fill-density (the exact "blank
  separator" signal that heuristics key on) and normalized row/column
  coordinates (CoordConv-style; tables cluster toward A1).
- ``AxialStripBlock`` (strip pooling à la Hou et al. 2020, applied to cell
  grids): each position receives a learned summary of its *entire row and
  entire column* — global context along exactly the two axes that matter in
  a grid, far cheaper than attention.

Both are known CV techniques individually; their combination on cell-feature
tensors is absent from TableSense.
"""

import torch
from torch import nn

# Number of channels DerivedChannels prepends.
NUM_DERIVED_CHANNELS = 4


class DerivedChannels(nn.Module):
    """Append row/col fill-density and normalized coordinates to the input.

    ``is_empty_index`` locates the emptiness channel in the featurization
    (channel 0 in this project's FEATURE_NAMES).
    """

    def __init__(self, is_empty_index: int = 0):
        super().__init__()
        self.is_empty_index = is_empty_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, H, W)
        batch, _, height, width = x.shape
        non_empty = 1.0 - x[:, self.is_empty_index : self.is_empty_index + 1]
        row_density = non_empty.mean(dim=3, keepdim=True).expand(-1, -1, -1, width)
        col_density = non_empty.mean(dim=2, keepdim=True).expand(-1, -1, height, -1)
        rows = torch.linspace(0, 1, height, device=x.device, dtype=x.dtype)
        cols = torch.linspace(0, 1, width, device=x.device, dtype=x.dtype)
        row_coord = rows.view(1, 1, height, 1).expand(batch, 1, height, width)
        col_coord = cols.view(1, 1, 1, width).expand(batch, 1, height, width)
        return torch.cat([x, row_density, col_density, row_coord, col_coord], dim=1)


class AxialStripBlock(nn.Module):
    """Residual strip-pooling: fuse whole-row and whole-column summaries."""

    def __init__(self, channels: int, kernel_size: int = 7, norm_groups: int = 8):
        super().__init__()
        padding = kernel_size // 2
        # Row branch: pool width away, convolve along H, broadcast back.
        self.row_conv = nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1),
                                  padding=(padding, 0))
        # Column branch: pool height away, convolve along W, broadcast back.
        self.col_conv = nn.Conv2d(channels, channels, kernel_size=(1, kernel_size),
                                  padding=(0, padding))
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GroupNorm(norm_groups, channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, H, W)
        _, _, height, width = x.shape
        row_summary = self.row_conv(x.mean(dim=3, keepdim=True)).expand(-1, -1, -1, width)
        col_summary = self.col_conv(x.mean(dim=2, keepdim=True)).expand(-1, -1, height, -1)
        context = self.fuse(row_summary + col_summary)
        return self.activation(x + context)
