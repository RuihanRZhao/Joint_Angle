import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class StudentModel(nn.Module):
    """
    Lightweight student model for joint human segmentation and pose estimation.
    Uses MobileNetV3 as the feature extractor and small decoder heads.
    """
    def __init__(self, num_keypoints: int, seg_channels: int = 1, width_mult: float = 1.0, pretrained_backbone: bool = True):
        super(StudentModel, self).__init__()
        # Load a MobileNetV3 backbone (large) up to the final convolution layer
        backbone_model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained_backbone else None)
        self.backbone = backbone_model.features  # Sequential backbone (outputs 1280-channel feature)
        # Identify channels at key feature points for skip connections
        # MobileNetV3-Large feature map channels at:
        # - 1/4 resolution (stride 4) -> 24 channels
        # - 1/8 resolution (stride 8) -> 40 channels
        # - 1/16 resolution (stride 16) -> 112 channels
        # - final 1/32 resolution -> 1280 channels
        self.skip_index_1 = 6   # index in backbone where feature is 1/8 res (approx 40 channels)
        self.skip_index_2 = 12  # index where feature is 1/16 res (approx 112 channels)

        # Convolution layers to reduce channels for skip features and backbone output
        self.conv_reduce_low = nn.Sequential(
            nn.Conv2d(40, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )  # reduce 1/8 res feature (approx 40 ch) to 64
        self.conv_reduce_mid = nn.Sequential(
            nn.Conv2d(112, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )  # reduce 1/16 res feature (112 ch) to 64
        self.conv_reduce_high = nn.Sequential(
            nn.Conv2d(1280, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )  # reduce final 1/32 res feature (1280 ch) to 64

        # Decoder convolution blocks to fuse features
        self.conv_fuse_mid = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )  # fuse 1/16 feature with upsampled 1/32
        self.conv_fuse_low = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )  # fuse 1/8 feature with upsampled 1/16

        # Output heads for segmentation and pose
        self.seg_head = nn.Conv2d(64, seg_channels, kernel_size=1)        # segmentation logits output (1 channel for binary mask)
        self.pose_head = nn.Conv2d(64, num_keypoints, kernel_size=1)      # pose heatmap output (one channel per keypoint)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        B, C, H, W = x.shape
        # Backbone forward with manual capture of intermediate features
        feat_low = None   # 1/8 resolution feature
        feat_mid = None   # 1/16 resolution feature
        feat_high = None  # 1/32 resolution (final backbone output)
        out = x
        for i, layer in enumerate(self.backbone):
            out = layer(out)
            if i == self.skip_index_1:
                feat_low = out               # feature at 1/8 resolution
            if i == self.skip_index_2:
                feat_mid = out              # feature at 1/16 resolution
        feat_high = out                      # final feature at 1/32 resolution

        # Reduce channel dimensions of features
        low = self.conv_reduce_low(feat_low)       # -> 64 channels at 1/8 res
        mid = self.conv_reduce_mid(feat_mid)       # -> 64 channels at 1/16 res
        high = self.conv_reduce_high(feat_high)    # -> 64 channels at 1/32 res

        # Upsample high-level feature to 1/16 res and fuse with mid-level
        high_up = F.interpolate(high, size=mid.shape[2:], mode='bilinear', align_corners=False)
        mid_fused = self.conv_fuse_mid(high_up + mid)  # combine high and mid features (64 channels at 1/16 res)

        # Upsample fused mid feature to 1/8 res and fuse with low-level
        mid_up = F.interpolate(mid_fused, size=low.shape[2:], mode='bilinear', align_corners=False)
        low_fused = self.conv_fuse_low(mid_up + low)    # combine with low features (64 channels at 1/8 res)

        # Segmentation output: upsample to input resolution
        seg_logits_small = self.seg_head(low_fused)     # raw logits at 1/8 res
        seg_logits = F.interpolate(seg_logits_small, size=(H, W), mode='bilinear', align_corners=False)

        # Pose output: upsample one more time to 1/4 resolution for finer keypoint localization
        pose_feat = F.interpolate(low_fused, scale_factor=2.0, mode='bilinear', align_corners=False)
        pose_logits = self.pose_head(pose_feat)         # raw heatmap logits at ~1/4 of input size

        if not return_features:
            return seg_logits, pose_logits
        # If features requested, return the fused feature maps as well for distillation
        return seg_logits, pose_logits, low_fused, pose_feat
