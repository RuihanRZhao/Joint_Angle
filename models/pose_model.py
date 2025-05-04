"""
pose.py: Defines the enhanced pose estimation network.
The network uses a ResNet backbone (e.g., ResNet-50) and deconvolutional layers
to generate heatmaps for keypoint estimation.
"""


import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class PoseEstimationModel(nn.Module):
    def __init__(self, in_channels: int = 3, num_keypoints: int = 17, num_pafs: int = 32, num_deconv_layers: int = 3):
        super().__init__()
        # Backbone: ResNet50
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # up to conv5_x

        # Deconvolutional layers to upsample features
        self.deconv_layers = self._make_deconv_layers(num_deconv_layers)
        # Head: heatmaps for keypoints
        self.heatmap_head = nn.Conv2d(256, num_keypoints, kernel_size=1)
        # Head: Part Affinity Fields for limbs
        self.paf_head = nn.Conv2d(256, num_pafs, kernel_size=1)

    def _make_deconv_layers(self, num_layers: int):
        layers = []
        in_channels = 2048
        for _ in range(num_layers):
            layers.append(nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(256))
            layers.append(nn.ReLU(inplace=True))
            in_channels = 256
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)            # [B, 2048, H/32, W/32]
        up_feats = self.deconv_layers(feats)  # [B, 256, H/4, W/4]（取决于 deconv 层数）
        heatmaps = self.heatmap_head(up_feats)  # [B, K, H/4, W/4]
        pafs = self.paf_head(up_feats)          # [B, 2*L, H/4, W/4]
        return heatmaps, pafs