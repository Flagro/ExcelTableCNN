import torch.nn as nn
import torchvision.models as models

class FCNBackbone(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        # self.conv1 = nn.Conv2d(num_features, 64, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # Increasing to 256 channels
        vgg16_features = models.vgg16(pretrained=True).features[:-1] # Use VGG16's features up to the second to last layer
        
        # Replace VGG16's first convolution layer with one that accepts num_features channels
        vgg16_features[0] = nn.Conv2d(num_features, 64, kernel_size=3, padding=1)
        self.vgg16_features = vgg16_features
        self.out_channels = 512  # Set the correct number of output channels

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        x = self.vgg16_features(x)  # Apply VGG16 features
        # Assume x is shaped (N, 256, H', W') where H' and W' are downscaled from the original H and W.
        return x  # The features are now ready for the RPN
