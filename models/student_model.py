import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class StudentModel(nn.Module):
    """修改后的学生模型，支持分割结果作为姿态估计的输入"""

    def __init__(self, num_keypoints: int, seg_channels: int = 1, width_mult: float = 1.0,
                 pretrained_backbone: bool = True):
        super(StudentModel, self).__init__()
        # 使用MobileNetV3作为轻量级特征提取器
        backbone_model = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained_backbone else None)
        self.backbone = backbone_model.features

        # 特征点索引，用于跳跃连接
        self.skip_index_1 = 6  # 1/8分辨率特征点
        self.skip_index_2 = 12  # 1/16分辨率特征点

        # 特征降维卷积
        self.conv_reduce_low = nn.Sequential(
            nn.Conv2d(40, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )

        self.conv_reduce_mid = nn.Sequential(
            nn.Conv2d(112, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )

        self.conv_reduce_high = nn.Sequential(
            nn.Conv2d(1280, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )

        # 特征融合卷积
        self.conv_fuse_mid = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )

        self.conv_fuse_low = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )

        # 分割输出头
        self.seg_head = nn.Conv2d(64, seg_channels, kernel_size=1)

        # 新增：处理分割结果与特征融合的卷积层
        self.seg_to_pose_conv = nn.Sequential(
            nn.Conv2d(64 + seg_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 姿态估计输出头
        self.pose_head = nn.Conv2d(64, num_keypoints, kernel_size=1)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        B, C, H, W = x.shape

        # 特征提取
        feat_low = None
        feat_mid = None
        feat_high = None

        out = x
        for i, layer in enumerate(self.backbone):
            out = layer(out)
            if i == self.skip_index_1:
                feat_low = out
            if i == self.skip_index_2:
                feat_mid = out
        feat_high = out

        # 特征降维和融合
        low = self.conv_reduce_low(feat_low)
        mid = self.conv_reduce_mid(feat_mid)
        high = self.conv_reduce_high(feat_high)

        # 上采样高级特征并与中级特征融合
        high_up = F.interpolate(high, size=mid.shape[2:], mode='bilinear', align_corners=False)
        mid_fused = self.conv_fuse_mid(high_up + mid)

        # 上采样融合特征并与低级特征融合
        mid_up = F.interpolate(mid_fused, size=low.shape[2:], mode='bilinear', align_corners=False)
        low_fused = self.conv_fuse_low(mid_up + low)

        # 分割输出
        seg_logits_small = self.seg_head(low_fused)
        seg_logits = F.interpolate(seg_logits_small, size=(H, W), mode='bilinear', align_corners=False)

        # 关键修改：使用分割结果作为姿态估计的输入
        seg_probs_small = torch.sigmoid(seg_logits_small)  # 将logits转换为概率

        # 上采样特征至更高分辨率用于姿态估计
        pose_feat = F.interpolate(low_fused, scale_factor=2.0, mode='bilinear', align_corners=False)
        seg_up = F.interpolate(seg_probs_small, size=pose_feat.shape[2:], mode='bilinear', align_corners=False)

        # 将分割结果与特征融合
        combined_feat = torch.cat([pose_feat, seg_up], dim=1)
        pose_feat_refined = self.seg_to_pose_conv(combined_feat)

        # 姿态估计输出
        pose_logits = self.pose_head(pose_feat_refined)

        if not return_features:
            return seg_logits, pose_logits

        # 如果需要返回特征（用于蒸馏）
        return seg_logits, pose_logits, low_fused, pose_feat_refined
