"""Faster R-CNN customized for spreadsheet tensors.

Differences from the stock torchvision detector:
- anchors sized in cell units for table shapes (few-cell tables up to
  tall multi-hundred-row ones);
- the input transform skips resizing and normalization: a sheet tensor is
  not a natural image — channels are binary features and coordinates are
  cells, so rescaling would destroy cell alignment.
"""

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign

# Tuned on the 2,613 annotated TableSense/VEnron2 ground-truth boxes via
# excel_table_cnn.data.census (2026-07): width p50=8 p95=31; height p50=21
# p95=167; h/w ratio p50=2.5 p95=26 — spreadsheet tables are tall. This
# 8x9 lattice covers 99.5% of GT boxes at IoU>=0.5 and 71.3% at IoU>=0.7
# (previous 7x7 lattice with ratios capped at 10: 93.5% / 45.2%).
DEFAULT_ANCHOR_SIZES = ((3, 5, 8, 13, 21, 34, 64, 128),)
DEFAULT_ASPECT_RATIOS = ((0.15, 0.35, 0.7, 1.4, 2.8, 5.5, 11.0, 22.0, 45.0),)


class SkipTransform(GeneralizedRCNNTransform):
    """Batches images (padding only) but performs no resize/normalize."""

    def normalize(self, image):
        return image

    def resize(self, image, target):
        return image, target


class CustomFasterRCNN(FasterRCNN):
    def __init__(
        self,
        backbone,
        in_channels: int,
        num_classes: int = 2,
        anchor_sizes=DEFAULT_ANCHOR_SIZES,
        aspect_ratios=DEFAULT_ASPECT_RATIOS,
        box_score_thresh: float = 0.5,
    ):
        anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
        # 14x14 pooling per the paper — finer boundary evidence per RoI.
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=["0"], output_size=14, sampling_ratio=2
        )
        super().__init__(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            rpn_pre_nms_top_n=dict(training=2000, testing=1000),
            rpn_post_nms_top_n=dict(training=2000, testing=1000),
            box_roi_pool=roi_pooler,
            box_score_thresh=box_score_thresh,
        )
        # min/max size and mean/std are unused because SkipTransform bypasses
        # resize() and normalize(); they only satisfy the constructor.
        self.transform = SkipTransform(
            min_size=1,
            max_size=100000,
            image_mean=[0.0] * in_channels,
            image_std=[1.0] * in_channels,
        )
