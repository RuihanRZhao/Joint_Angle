import torch
import torch.nn as nn
import torchvision.models as models

class PoseEstimationModel(nn.Module):
    """ResNet backbone with deconvolutional head for keypoint heatmap prediction."""
    def __init__(self, backbone="resnet50", num_keypoints=17):
        super().__init__()
        self.num_keypoints = num_keypoints
        # Load ResNet backbone pre-trained
        resnet = getattr(models, backbone)(pretrained=True)
        # Use all layers up to the final conv layer (exclude avgpool and fc)
        # For ResNet50, final conv layer output is 2048 channels (stride 32 downsampled from input)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        # Deconvolutional head: 3 deconv layers to upsample from stride 32 to stride 4 (2x2x2 upsampling)
        # Each deconv layer outputs 256 channels
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(2048 if backbone in ["resnet50", "resnet101", "resnet152"] else 512,
                               256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        # Final prediction conv layer: 1x1 conv to get one heatmap per keypoint
        self.final_layer = nn.Conv2d(256, num_keypoints, kernel_size=1)

    def forward(self, x):
        # x shape: [B,3,H,W]
        feat = self.backbone(x)
        out = self.deconv_layers(feat)
        out = self.final_layer(out)  # shape: [B, num_keypoints, H/4, W/4] assuming input H,W multiple of 32
        return out
