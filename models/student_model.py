# optimized_seg_pose_mbv2.py
# Using MobileNetV2 backbone for lightweight segmentation + PFLD for pose, sharing backbone for efficiency

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# -----------------------------------
# MobileNetV2-based Segmentation
# -----------------------------------
class MobileNetV2Seg(nn.Module):
    def __init__(self, num_classes=1, width_mult=1.0, pretrained_backbone=True):
        super().__init__()
        # Backbone: MobileNetV2
        mb2 = mobilenet_v2(
            weights=MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained_backbone else None,
            width_mult=width_mult
        )
        self.features = mb2.features  # feature extractor
        in_ch = mb2.last_channel     # typically 1280 * width_mult
        # Segmentation head: simple decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 2, num_classes, kernel_size=1)
        )

    def forward(self, x):
        feats = self.features(x)           # [B, C, H/32, W/32]
        up = F.interpolate(feats, scale_factor=32, mode='bilinear', align_corners=False)
        seg_logits = self.decoder(up)      # [B,1,H,W]
        return seg_logits

# -----------------------------------
# PFLD Pose Estimation (Fixed)
# -----------------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k, s=1, p=0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.act(self.dw_bn(self.dw(x)))
        x = self.act(self.pw_bn(self.pw(x)))
        return x

class PFLD(nn.Module):
    def __init__(self, num_keypoints=17, width_mult=1.0):
        super().__init__()
        base_ch = int(64 * width_mult)
        self.conv1 = ConvBNReLU(3, base_ch, 3, 2, 1)
        self.conv2 = DepthwiseSeparable(base_ch, base_ch * 2, 2)
        self.conv3 = DepthwiseSeparable(base_ch * 2, base_ch * 4, 2)
        self.conv4 = DepthwiseSeparable(base_ch * 4, base_ch * 4, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_ch * 4, num_keypoints * 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avgpool(x).view(x.size(0), -1)
        out = self.fc(x)
        coords = out.view(-1, out.size(1) // 2, 2)
        return coords

# -----------------------------------
# Shared Backbone Multi-Task Model
# -----------------------------------
class StudentModel(nn.Module):
    def __init__(self, seg_width_mult=1.0, pose_width_mult=1.0):
        super().__init__()
        mb2 = mobilenet_v2(
            weights=MobileNet_V2_Weights.IMAGENET1K_V1,
            width_mult=seg_width_mult
        )
        self.features = mb2.features
        feat_ch = mb2.last_channel
        # Seg head
        self.seg_head = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(feat_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch // 2, 1, 1)
        )
        # Pose head (PFLD)
        self.pose_net = PFLD(num_keypoints=17, width_mult=pose_width_mult)

    def forward(self, x):
        feats = self.features(x)
        # Segmentation
        seg_up = F.interpolate(feats, scale_factor=32, mode='bilinear', align_corners=False)
        seg_logits = self.seg_head(seg_up)
        # Pose input: combine image and seg prob
        seg_prob = torch.sigmoid(seg_logits)
        pose_input = torch.cat([x, seg_prob], dim=1)
        keypoints = self.pose_net(pose_input)
        return seg_logits, keypoints
