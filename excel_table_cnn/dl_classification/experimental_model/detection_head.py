from torch import nn
from torch.nn import functional as F


class DetectionHead(nn.Module):
    def __init__(self, in_channels=64, pool_size=7, hidden_dim=128, debug=False):
        super().__init__()
        self.pool_size = pool_size
        fc_in_dim = in_channels * pool_size * pool_size

        self.fc1 = nn.Linear(fc_in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.cls_logits = nn.Linear(hidden_dim, 2)
        self.bbox_pred = nn.Linear(hidden_dim, 4)
        self.pbr_pred = nn.Linear(hidden_dim, 4)
        self.debug = debug

    def forward(self, roi_features):
        N, C, H, W = roi_features.shape
        x = roi_features.view(N, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        cls_logit = self.cls_logits(x)  # (N,2)
        bbox_delta = self.bbox_pred(x)  # (N,4)
        pbr_delta = self.pbr_pred(x)  # (N,4)

        if self.debug:
            print(
                f"[DetectionHead] cls_logit.shape={cls_logit.shape}, "
                f"bbox_delta[:3]={bbox_delta[:3].data}, pbr_delta[:3]={pbr_delta[:3].data}"
            )

        return cls_logit, bbox_delta, pbr_delta
