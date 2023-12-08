from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork


def get_anchor_generator():
    # Use predefined sizes and aspect ratios. The sizes should be tuples of (min, max).
    sizes = ((8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096),) # One sub-tuple for one feature level
    aspect_ratios = ((0.00390625, 0.0078125, 0.015625, 
                      0.03125, 0.0625, 0.125, 
                      0.25, 0.5, 1.0, 2.0, 
                      4.0, 8.0, 16.0, 32.0, 
                      64.0, 128.0, 256.0),)
    return AnchorGenerator(sizes=sizes, aspect_ratios=aspect_ratios)


# Create a lightweight transform class that overrides the normalize and resize methods
class SkipTransform(GeneralizedRCNNTransform):
    def __init__(self, min_size, max_size, image_mean, image_std):
        super().__init__(min_size, max_size, image_mean, image_std)
    
    def normalize(self, image):
        # Skip normalization or do your custom normalization if needed
        return image
    
    def resize(self, image, target):
        # Skip resizing or do your custom resize if needed
        # Returning the original image and target as is
        return image, target


# Define the custom Faster R-CNN class
class CustomFasterRCNN(FasterRCNN):
    def __init__(self, backbone, num_classes):
        # Initialize the RPN (Region Proposal Network)

        anchor_generator = get_anchor_generator()
        rpn_head = RPNHead(512, anchor_generator.num_anchors_per_location()[0])
        
        # Define the box ROI Pooler using RoIAlign
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=['0'],  # Assuming '0' is the name of the feature map to be used for RoI Align
            output_size=7,
            sampling_ratio=2
        )
        
        # Instantiate the Faster R-CNN model using our backbone, our custom RPN, and RoI Align
        super().__init__(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            rpn_head=rpn_head,
            rpn_fg_iou_thresh=0.7,
            rpn_bg_iou_thresh=0.3,
            rpn_batch_size_per_image=256,
            rpn_positive_fraction=0.5,
            rpn_pre_nms_top_n=dict(training=2000, testing=1000),
            rpn_post_nms_top_n=dict(training=2000, testing=1000),
            rpn_nms_thresh=0.7,
            box_roi_pool=roi_pooler,
        )

        # Replace the pre-trained head with a new one (number of classes is different)
        in_features = self.roi_heads.box_predictor.cls_score.in_features
        self.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Set the custom skip transform as the transform for the CustomFasterRCNN with the dummy values
        min_size, max_size, image_mean, image_std = 100, 1333, ([0.485] * 17), ([0.229] * 17)
        self.transform = SkipTransform(min_size, max_size, image_mean, image_std)
