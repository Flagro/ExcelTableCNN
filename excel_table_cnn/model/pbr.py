"""Precise boundary refinement (PBR) — TableSense-inspired, discretized.

The paper's PBR observes that generic box regression cannot deliver
cell-exact boundaries and refines each edge from features local to that
edge, with a loss that plateaus beyond a tolerance of k cells. This module
keeps that structure — per-edge refinement, band-shaped receptive fields,
hard k-cell tolerance — with two deliberate differences:

- the offset is **classified over the 2k+1 integer positions** instead of
  regressed: cell boundaries are discrete, and argmax-snapping directly
  optimizes EoB-0 (exact-boundary accuracy);
- it is trained on **jittered ground-truth boxes** (each edge displaced by
  Uniform{-k..k}) rather than inside the RoI head, which teaches the same
  skill — recover the true boundary from a near-miss — without surgery on
  torchvision internals. Offsets beyond ±k cannot occur in training, which
  is exactly the paper's plateau: no gradient outside the tolerance window.
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import roi_align

EDGE_NAMES = ("left", "top", "right", "bottom")  # == box coordinate order


def jitter_boxes(
    boxes: torch.Tensor, k: int, height: int, width: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Displace every edge by Uniform{-k..k}, keeping boxes valid.

    Returns (jittered_boxes, offsets) where ``offsets = true - jittered``
    per edge — the classification targets, guaranteed within [-k, k].
    """
    if boxes.numel() == 0:
        return boxes, boxes
    jitter = torch.randint(-k, k + 1, boxes.shape, device=boxes.device).float()
    jittered = boxes + jitter
    # Keep at least one cell of extent and stay on the sheet.
    jittered[:, 0] = jittered[:, 0].clamp(0, width - 1).minimum(boxes[:, 2] - 1)
    jittered[:, 1] = jittered[:, 1].clamp(0, height - 1).minimum(boxes[:, 3] - 1)
    jittered[:, 2] = jittered[:, 2].clamp(1, width).maximum(jittered[:, 0] + 1)
    jittered[:, 3] = jittered[:, 3].clamp(1, height).maximum(jittered[:, 1] + 1)
    offsets = (boxes - jittered).clamp(-k, k)
    return jittered, offsets


class PBRHead(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        k: int = 7,
        band_pool: int = 7,
        trunk_channels: int = 64,
        hidden: int = 128,
    ):
        super().__init__()
        self.k = k
        self.band_pool = band_pool
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, trunk_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(trunk_channels, trunk_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # One classifier per edge type: the band around a left edge (table to
        # the right) looks nothing like the band around a right edge.
        flat = trunk_channels * (2 * k + 1)
        self.edge_heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(flat, hidden), nn.ReLU(inplace=True),
                nn.Linear(hidden, 2 * k + 1),
            )
            for _ in EDGE_NAMES
        )

    def _band_rois(self, boxes: torch.Tensor, edge: int) -> torch.Tensor:
        """Band region covering cells e-k .. e+k around the given edge, with
        full box extent along the other axis."""
        k = float(self.k)
        x1, y1, x2, y2 = boxes.unbind(dim=1)
        edge_coord = boxes[:, edge]
        if edge in (0, 2):  # vertical edge: band of columns
            rois = torch.stack([edge_coord - k, y1, edge_coord + k + 1, y2], dim=1)
        else:  # horizontal edge: band of rows
            rois = torch.stack([x1, edge_coord - k, x2, edge_coord + k + 1], dim=1)
        batch_idx = torch.zeros((len(boxes), 1), device=boxes.device)
        return torch.cat([batch_idx, rois], dim=1)

    def forward(self, features: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """features: (1, C, H, W) of ONE sheet; boxes: (N, 4).
        Returns logits (N, 4 edges, 2k+1 offsets)."""
        width = 2 * self.k + 1
        logits = []
        for edge in range(4):
            rois = self._band_rois(boxes, edge)
            if edge in (0, 2):
                band = roi_align(features, rois, output_size=(self.band_pool, width),
                                 aligned=True)
            else:
                band = roi_align(features, rois, output_size=(width, self.band_pool),
                                 aligned=True)
                band = band.permute(0, 1, 3, 2)  # offset axis last
            band = self.trunk(band)
            band = band.mean(dim=2)  # pool over the box-extent axis -> (N, C, width)
            logits.append(self.edge_heads[edge](band.flatten(1)))
        return torch.stack(logits, dim=1)  # (N, 4, 2k+1)

    def loss(self, features: torch.Tensor, gt_boxes: torch.Tensor,
             height: int, width: int, jitter_repeats: int = 4) -> torch.Tensor:
        """Jitter GT boxes, predict the recovery offsets, cross-entropy.

        Features are detached: the detection losses already shape the
        backbone, and a stationary feature space lets the snapping head
        converge orders of magnitude faster. Each GT box is jittered
        ``jitter_repeats`` times per step for denser supervision (bands are
        tiny, so this is nearly free).
        """
        if gt_boxes.numel() == 0:
            return features.sum() * 0.0
        features = features.detach()
        repeated = gt_boxes.repeat(jitter_repeats, 1)
        jittered, offsets = jitter_boxes(repeated, self.k, height, width)
        logits = self(features, jittered)  # (N*J, 4, 2k+1)
        targets = (offsets + self.k).long()
        return F.cross_entropy(logits.reshape(-1, 2 * self.k + 1), targets.reshape(-1))

    def refine(self, features: torch.Tensor, boxes: torch.Tensor,
               height: int, width: int) -> torch.Tensor:
        """Snap detector boxes to cell-exact boundaries (one pass).

        Boxes are grid-rounded first so bands are cell-aligned like in
        training; boxes that would degenerate keep their rounded coordinates.
        """
        if boxes.numel() == 0:
            return boxes
        rounded = boxes.round()
        rounded[:, 2] = rounded[:, 2].maximum(rounded[:, 0] + 1)
        rounded[:, 3] = rounded[:, 3].maximum(rounded[:, 1] + 1)
        logits = self(features, rounded)
        offsets = logits.argmax(dim=2).float() - self.k
        refined = rounded + offsets
        refined[:, 0::2] = refined[:, 0::2].clamp(0, width)
        refined[:, 1::2] = refined[:, 1::2].clamp(0, height)
        valid = (refined[:, 2] > refined[:, 0]) & (refined[:, 3] > refined[:, 1])
        return torch.where(valid.unsqueeze(1), refined, rounded)
