import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import InvertedResidual, GhostBottleneck

class JointPoseNet(nn.Module):
    def __init__(self, num_joints=17, input_size=(384, 216), bins=4):
        super(JointPoseNet, self).__init__()
        self.num_joints = num_joints
        self.input_w, self.input_h = input_size
        self.bins = bins
        self.out_w = self.input_w // 4
        self.out_h = self.input_h // 4
        self.x_classes = self.out_w * bins
        self.y_classes = self.out_h * bins

        # Backbone (MobileNetV2-like)
        self.backbone = nn.ModuleList()
        input_channels = 32
        self.backbone.append(nn.Sequential(
            nn.Conv2d(3, input_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU6(inplace=True)
        ))
        mobilenet_cfg = [
            (1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2),
            (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 1)
        ]
        in_ch = input_channels
        for t, c, n, s in mobilenet_cfg:
            for i in range(n):
                stride = s if i == 0 else 1
                use_se = c >= 64 and i == n - 1
                block = InvertedResidual(in_ch, c, stride, expand_ratio=t, use_se=use_se)
                self.backbone.append(block)
                in_ch = c

        # High-res branch
        self.highres_branch = nn.ModuleList([
            GhostBottleneck(24, 48, 24, 1, use_se=False),
            GhostBottleneck(24, 48, 32, 1, use_se=True),
            GhostBottleneck(32, 64, 32, 1, use_se=False)
        ])

        # Upsample
        self.upsample1 = nn.Upsample(scale_factor=2)  # 1/16 -> 1/8
        self.upsample2 = nn.Upsample(scale_factor=2)  # 1/8 -> 1/4

        # Feature fusion output
        self.fuse_conv = nn.Conv2d(160, 64, kernel_size=3, padding=1)

        # SimCC output heads
        self.keypoint_x_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_joints * self.x_classes, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.keypoint_y_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_joints * self.x_classes, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        features = x
        highres_input = None

        for idx, layer in enumerate(self.backbone):
            features = layer(features)
            if idx == 3:
                highres_input = features

        up1 = self.upsample1(features)
        up2 = self.upsample2(up1)  # [B, C, H/4, W/4]
        fused = self.fuse_conv(up2)

        # Classification logits
        B = x.size(0)
        out_x = self.keypoint_x_head(fused).view(B, self.num_joints, self.x_classes)
        out_y = self.keypoint_y_head(fused).view(B, self.num_joints, self.y_classes)

        # Softmax decoding
        prob_x = F.softmax(out_x, dim=2)
        prob_y = F.softmax(out_y, dim=2)

        # Compute expected coordinates (soft-argmax)
        device = x.device
        index_x = torch.arange(self.x_classes, device=device).float().view(1, 1, -1)
        index_y = torch.arange(self.y_classes, device=device).float().view(1, 1, -1)
        coord_x = (prob_x * index_x).sum(dim=2) * (self.input_w / self.x_classes)
        coord_y = (prob_y * index_y).sum(dim=2) * (self.input_h / self.y_classes)
        keypoints = torch.stack([coord_x, coord_y], dim=2)  # [B, K, 2]

        return out_x, out_y, keypoints