import torch.nn as nn
import torchvision.models as models


# Define the FCN model
class FCNBackbone(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.conv1 = nn.Conv2d(num_features, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=3, padding=1)  # Reducing to 3 channels
        self.features = models.vgg16(pretrained=True).features  # Use VGG16's features

    def forward(self, x):
        x = self.conv1(x)  # Convert to 3 channels
        x = self.features(x)  # Apply VGG16 features
        return x
