import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN

# Define the FCN model
class FCN(nn.Module):
    def __init__(self, num_features):
        super(FCN, self).__init__()
        self.conv1 = nn.Conv2d(num_features, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=3, padding=1)  # Reducing to 3 channels

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        return x

# Define the combined FCN + R-CNN model
class FCN_RCNN(nn.Module):
    def __init__(self, num_features):
        super(FCN_RCNN, self).__init__()
        self.fcn = FCN(num_features)

        # Create a backbone from the FCN
        backbone = BackboneWithFPN(self.fcn, return_layers={self.fcn.conv3: "0"}, out_channels=3)
        
        # RPN (Region Proposal Network) and Anchor Generator
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),) * 5)

        # Create the Faster R-CNN model using the custom backbone
        self.rcnn = FasterRCNN(backbone, num_classes=2, rpn_anchor_generator=anchor_generator)

    def forward(self, images):
        # Forward pass through FCN
        features = self.fcn(images)

        # Forward pass through R-CNN
        output = self.rcnn(features)
        return output
