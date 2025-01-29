from torch import nn
from torch.nn import functional as F


class RPN(nn.Module):
    def __init__(self, in_channels=64, num_anchors=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.obj_score = nn.Conv2d(in_channels, num_anchors, 1)
        self.bbox_deltas = nn.Conv2d(in_channels, num_anchors * 4, 1)

    def forward(self, feat):
        x = F.relu(self.conv(feat))
        scores = self.obj_score(x)  # (B,num_anchors,H,W)
        deltas = self.bbox_deltas(x)  # (B,num_anchors*4,H,W)
        return scores, deltas
