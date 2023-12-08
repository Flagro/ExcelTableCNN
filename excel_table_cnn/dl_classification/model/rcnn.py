from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from .rpn import RPN, get_rpn_params


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
        rpn = RPN()
        rpn_params = get_rpn_params()
        
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
            # rpn=rpn,
            # rpn_anchor_generator=rpn.anchor_generator,
            box_roi_pool=roi_pooler,
            **rpn_params
        )

        # Replace the pre-trained head with a new one (number of classes is different)
        in_features = self.roi_heads.box_predictor.cls_score.in_features
        self.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Set the custom skip transform as the transform for the CustomFasterRCNN
        # Choose dummy values for min_size and max_size as they won't have an effect
        # Choose dummy values for image_mean and image_std as they won't have an effect
        min_size = (100,)
        max_size = 1333
        image_mean = [0.0] * 17  # Adjust depending on your case
        image_std = [1.0] * 17   # Adjust depending on your case
        self.transform = SkipTransform(min_size, max_size, image_mean, image_std)
