"""The table detection model: FCN backbone + customized Faster R-CNN."""

import torch

from .backbone import FCNBackbone
from .rcnn import CustomFasterRCNN


class TableDetectionModel(torch.nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 2, **rcnn_kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.backbone = FCNBackbone(in_channels)
        self.model = CustomFasterRCNN(
            self.backbone, in_channels=in_channels, num_classes=num_classes,
            **rcnn_kwargs,
        )

    def forward(self, images, targets=None):
        """Training mode (with targets): dict of losses. Eval mode: list of
        {"boxes", "labels", "scores"} per image."""
        return self.model(images, targets)


def build_model(in_channels: int, num_classes: int = 2, **rcnn_kwargs) -> TableDetectionModel:
    return TableDetectionModel(in_channels, num_classes=num_classes, **rcnn_kwargs)
