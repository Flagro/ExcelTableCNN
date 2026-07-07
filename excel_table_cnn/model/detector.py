"""The table detection model: grid-context FCN backbone + customized Faster
R-CNN + PBR boundary snapping."""

import torch

from .backbone import FCNBackbone
from .pbr import PBRHead
from .rcnn import CustomFasterRCNN


class TableDetectionModel(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 2,
        use_pbr: bool = True,
        pbr_k: int = 7,
        use_grid_context: bool = True,
        **rcnn_kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.use_grid_context = use_grid_context
        self.pbr_k = pbr_k
        self.backbone = FCNBackbone(in_channels, use_grid_context=use_grid_context)
        self.model = CustomFasterRCNN(
            self.backbone, in_channels=in_channels, num_classes=num_classes,
            **rcnn_kwargs,
        )
        self.pbr = PBRHead(self.backbone.out_channels, k=pbr_k) if use_pbr else None

    @property
    def config(self) -> dict:
        """Architecture flags needed to rebuild this model from a checkpoint."""
        return {
            "use_pbr": self.pbr is not None,
            "pbr_k": self.pbr_k,
            "use_grid_context": self.use_grid_context,
        }

    def forward(self, images, targets=None):
        """Training mode (with targets): dict of losses (incl. ``loss_pbr``
        when PBR is enabled). Eval mode: list of {"boxes", "labels",
        "scores"} per image, with boxes PBR-snapped to cell boundaries."""
        if self.training:
            losses = self.model(images, targets)
            if self.pbr is not None:
                pbr_losses = []
                for image, target in zip(images, targets):
                    features = self.backbone(image.unsqueeze(0))
                    _, height, width = image.shape
                    pbr_losses.append(
                        self.pbr.loss(features, target["boxes"], height, width)
                    )
                losses["loss_pbr"] = torch.stack(pbr_losses).mean()
            return losses

        detections = self.model(images)
        if self.pbr is not None:
            for image, detection in zip(images, detections):
                if len(detection["boxes"]) == 0:
                    continue
                features = self.backbone(image.unsqueeze(0))
                _, height, width = image.shape
                detection["boxes"] = self.pbr.refine(
                    features, detection["boxes"], height, width
                )
        return detections


def build_model(in_channels: int, num_classes: int = 2, **kwargs) -> TableDetectionModel:
    return TableDetectionModel(in_channels, num_classes=num_classes, **kwargs)
