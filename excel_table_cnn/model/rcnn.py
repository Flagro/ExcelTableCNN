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

DEFAULT_ANCHOR_SIZES = ((4, 8, 16, 32, 64, 128, 256),)
DEFAULT_ASPECT_RATIOS = ((0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0),)


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
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=["0"], output_size=7, sampling_ratio=2
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
