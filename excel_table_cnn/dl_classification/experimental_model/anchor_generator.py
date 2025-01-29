import math
import torch
import torch.nn as nn


class AnchorGenerator(nn.Module):
    def __init__(self, scales=None, ratios=None, stride=1, debug=False):
        super().__init__()
        if scales is None:
            # Example: powers of 2 from 8 to 4096
            scales = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        if ratios is None:
            # Some aspect ratios from 1/256 to 256
            ratios = [0.004, 0.0156, 0.0625, 0.25, 1.0, 4.0, 16.0, 64.0]
        self.scales = scales
        self.ratios = ratios
        self.stride = stride
        self.debug = debug

    def forward(self, feature_map):
        device = feature_map.device
        B, C, H, W = feature_map.shape

        base_anchors = []
        for s in self.scales:
            for r in self.ratios:
                w = math.sqrt(s * s * r)
                h = math.sqrt(s * s / r)
                x1 = -w / 2
                y1 = -h / 2
                x2 = w / 2
                y2 = h / 2
                base_anchors.append([x1, y1, x2, y2])
        base_anchors = torch.tensor(base_anchors, dtype=torch.float32, device=device)
        num_anchors = base_anchors.shape[0]  # e.g. len(scales)*len(ratios)

        # grid
        shifts_x = torch.arange(0, W, dtype=torch.float32, device=device)
        shifts_y = torch.arange(0, H, dtype=torch.float32, device=device)
        shift_x, shift_y = torch.meshgrid(shifts_x, shifts_y, indexing="xy")
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)

        XY = shift_x.shape[0]  # H*W
        base_expanded = base_anchors.unsqueeze(0).expand(XY, num_anchors, 4)
        shifts = (
            torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)
            .unsqueeze(1)
            .expand(XY, num_anchors, 4)
        )

        anchors = base_expanded + shifts  # shape (H*W, num_anchors, 4)
        anchors = anchors.reshape(-1, 4)

        if self.debug:
            print(
                f"[AnchorGenerator] #anchors = {anchors.shape[0]}, "
                f"min={anchors.min(dim=0).values}, max={anchors.max(dim=0).values}"
            )

        return anchors
