import torch
import torch.nn as nn
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork


def get_anchor_generator():
    # Use predefined sizes and aspect ratios. The sizes should be tuples of (min, max).
    sizes = ((8,), (16,), (32,), (64,), (128,), (256,), (512,), (1024,), (2048,), (4096,))
    aspect_ratios = ((0.00390625,), (0.0078125,), (0.015625,), (0.03125,), (0.0625,), (0.125,), (0.25,), (0.5,), (1.0,), (2.0,), (4.0,), (8.0,), (16.0,), (32.0,), (64.0,), (128.0,), (256.0,))
    return AnchorGenerator(sizes, aspect_ratios)


class RPN(nn.Module):
    def __init__(self):
        super().__init__()

        self.anchor_generator = get_anchor_generator()

        # Assuming that the backbone outputs a single feature map "feat1", with 512 channels
        self.head = RPNHead(512, self.anchor_generator.num_anchors_per_location()[0])
        
        # Predefined values for proposal matching, these can be fine tuned
        self.proposal_matcher = torch.nn.modules.Module()
        
        self.fg_bg_sampler = torch.nn.modules.Module()
        
        self.box_coder = torch.nn.modules.Module()
        
        # The RegionProposalNetwork brings together the anchor generator and the RPN head
        self.rpn = RegionProposalNetwork(
            self.anchor_generator, self.head,
            fg_iou_thresh=0.7, bg_iou_thresh=0.3,
            batch_size_per_image=256, positive_fraction=0.5,
            pre_nms_top_n=dict(training=2000, testing=1000),
            post_nms_top_n=dict(training=2000, testing=1000),
            nms_thresh=0.7,
        )

    def forward(self, images, features, targets=None):
        # Delegate to the internal RegionProposalNetwork the work
        proposals, proposal_losses = self.rpn(images, features, targets)

        return proposals, proposal_losses
