"""
pose.py: Defines the enhanced pose estimation network.
The network uses a ResNet backbone (e.g., ResNet-50) and deconvolutional layers
to generate heatmaps for keypoint estimation.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights

class PoseEstimationModel(nn.Module):
    def __init__(self, in_channels: int = 3, num_keypoints: int = 17):
        super().__init__()
        # Use a pretrained ResNet backbone (e.g., ResNet-50)
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Modify first conv layer to accept in_channels (e.g., 3 or 3+segmentation_channels)
        self.backbone = resnet
        if in_channels != 3:
            # Replace the first conv layer
            self.backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        # Remove fully connected and pool layers
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # up to conv5_x

        # Deconvolutional layers to upsample and refine heatmaps
        self.deconv_layers = self._make_deconv_layer()

        # Final layer to produce heatmaps (one per keypoint)
        self.final_layer = nn.Conv2d(
            in_channels=256, out_channels=num_keypoints, kernel_size=1, stride=1, padding=0
        )

    def _make_deconv_layer(self):
        layers = []
        # Example: 3 layers of deconv (like SimpleBaseline)
        layers.append(nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for pose estimation. Input x should have shape [B, in_channels, H, W].
        Returns heatmap predictions [B, num_keypoints, H_out, W_out] (typically downsampled).
        """
        features = self.backbone(x)       # Extract features
        heatmaps = self.deconv_layers(features)
        heatmaps = self.final_layer(heatmaps)
        return heatmaps
