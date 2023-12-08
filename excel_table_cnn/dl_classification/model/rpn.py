import torch
import torch.nn as nn
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork


def get_anchor_generator():
    # Use predefined sizes and aspect ratios. The sizes should be tuples of (min, max).
    sizes = ((8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096),) # One sub-tuple for one feature level
    aspect_ratios = ((0.125,),) # One sub-tuple matching the one feature level
    return AnchorGenerator(sizes=sizes, aspect_ratios=aspect_ratios)


def get_rpn_params(out_channels=512):
    anchor_generator = get_anchor_generator()
    rpn_head = RPNHead(out_channels, anchor_generator.num_anchors_per_location()[0])
    return {"anchor_generator": anchor_generator,
            "head": rpn_head,
            "fg_iou_thresh": 0.7,
            "bg_iou_thresh": 0.3,
            "batch_size_per_image": 256,
            "positive_fraction": 0.5,
            "pre_nms_top_n": dict(training=2000, testing=1000),
            "post_nms_top_n": dict(training=2000, testing=1000),
            "nms_thresh": 0.7}


class RPN(RegionProposalNetwork):
    def __init__(self, out_channels=512):
        anchor_generator = get_anchor_generator()

        # Assuming that the backbone outputs a single feature map "feat1", with 512 channels
        rpn_head = RPNHead(out_channels, anchor_generator.num_anchors_per_location()[0])
        
        # Predefined values for proposal matching, these can be fine tuned
        # self.proposal_matcher = torch.nn.modules.Module()
        
        # self.fg_bg_sampler = torch.nn.modules.Module()
        
        # self.box_coder = torch.nn.modules.Module()

        super().__init__(
            anchor_generator=anchor_generator,
            head=rpn_head,
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n=dict(training=2000, testing=1000),
            post_nms_top_n=dict(training=2000, testing=1000),
            nms_thresh=0.7
        )
