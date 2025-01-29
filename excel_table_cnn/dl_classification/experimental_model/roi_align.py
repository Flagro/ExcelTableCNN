import torch
from torch import nn
import torch.nn.functional as F


class RoIAlign(nn.Module):
    def __init__(self, out_size=7, debug=False):
        super().__init__()
        self.out_size = out_size
        self.debug = debug

    def forward(self, feature_map, proposals):
        device = feature_map.device
        B, C, H, W = feature_map.shape
        pooled = []
        for i in range(B):
            boxes = proposals[i]
            if len(boxes) == 0:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box
                # clamp
                x1c = max(0, min(x1.item(), W - 1))
                x2c = max(0, min(x2.item(), W - 1))
                y1c = max(0, min(y1.item(), H - 1))
                y2c = max(0, min(y2.item(), H - 1))
                roi = feature_map[
                    i, :, int(y1c) : int(y2c + 1), int(x1c) : int(x2c + 1)
                ]
                resized = F.interpolate(
                    roi.unsqueeze(0),
                    size=(self.out_size, self.out_size),
                    mode="bilinear",
                    align_corners=False,
                )
                pooled.append(resized.squeeze(0))
        if len(pooled) == 0:
            return torch.empty((0, C, self.out_size, self.out_size), device=device)
        out = torch.stack(pooled, dim=0)
        if self.debug:
            print(
                f"[RoIAlign] Pooled shape={out.shape}, min={out.min()}, max={out.max()}"
            )
        return out
