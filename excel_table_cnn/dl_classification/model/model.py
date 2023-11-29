import torch

from .fcn_backbone import FCNBackbone
from .rcnn import CustomFasterRCNN


# Define the main model
class TableDetectionModel(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Initialize the backbone, a modified FCN with an appropriate number of input channels
        self.backbone = FCNBackbone(17)

        # Pass the backbone and the number of classes to the CustomFasterRCNN
        self.model = CustomFasterRCNN(self.backbone, num_classes)

    def forward(self, images, targets=None):
        return self.model(images, targets)
