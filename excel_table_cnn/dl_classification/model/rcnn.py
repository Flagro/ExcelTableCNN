from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign

from .rpn import RPN


# Define the custom Faster R-CNN class
class CustomFasterRCNN(FasterRCNN):
    def __init__(self, backbone, num_classes):
        # Initialize the RPN (Region Proposal Network)
        rpn = RPN()
        
        # Define the box ROI Pooler using RoIAlign
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=['0'],  # Assuming '0' is the name of the feature map to be used for RoI Align
            output_size=7,
            sampling_ratio=2
        )
        
        # Instantiate the Faster R-CNN model using our backbone, our custom RPN, and RoI Align
        super(CustomFasterRCNN, self).__init__(
            backbone=backbone,
            num_classes=num_classes,
            rpn=rpn,
            box_roi_pool=roi_pooler,
        )

        # Replace the pre-trained head with a new one (number of classes is different)
        in_features = self.roi_heads.box_predictor.cls_score.in_features
        self.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
