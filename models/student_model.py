import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class StudentModel(nn.Module):
    """学生模型：轻量级特征提取 + 分割结果辅助姿态估计"""

    def __init__(self, num_keypoints: int, seg_channels: int = 1,
                 width_mult: float = 1.0, pretrained_backbone: bool = True):
        super(StudentModel, self).__init__()
        # 1. Backbone：MobileNetV3 Large
        backbone_model = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
            if pretrained_backbone else None,
            width_mult=width_mult
        )
        # 只保留 features 部分
        self.backbone = backbone_model.features

        # 动态获取最后一层特征图的通道数
        high_channels = backbone_model.features[-1].out_channels

        # 2. 跳跃连接索引（取 1/8 & 1/16 分辨率特征）
        self.skip_index_1 = 6
        self.skip_index_2 = 12

        # 3. 特征降维：low / mid / high
        #    low-level 特征（通道数 backbone.features[6].out_channels）
        self.conv_reduce_low = nn.Sequential(
            nn.Conv2d(backbone_model.features[self.skip_index_1].out_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        #    mid-level 特征（通道数 backbone.features[12].out_channels）
        self.conv_reduce_mid = nn.Sequential(
            nn.Conv2d(backbone_model.features[self.skip_index_2].out_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        #    high-level 特征（动态读取到的 high_channels）
        self.conv_reduce_high = nn.Sequential(
            nn.Conv2d(high_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 4. 融合卷积
        self.conv_fuse_mid = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv_fuse_low = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 5. 分割输出头
        self.seg_head = nn.Conv2d(64, seg_channels, kernel_size=1)

        # 6. 分割结果 + 特征融合，用于姿态分支
        #    输入通道 = 64（特征） + seg_channels（概率图）
        self.seg_to_pose_conv = nn.Sequential(
            nn.Conv2d(64 + seg_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # 姿态输出头
        self.pose_head = nn.Conv2d(64, num_keypoints, kernel_size=1)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        B, C, H, W = x.shape

        # 1. Backbone 前向，记录 multi-scale 特征
        feat_low = feat_mid = feat_high = None
        out = x
        for idx, layer in enumerate(self.backbone):
            out = layer(out)
            if idx == self.skip_index_1:
                feat_low = out
            if idx == self.skip_index_2:
                feat_mid = out
        feat_high = out  # 最后一层输出

        # 2. 降维
        low = self.conv_reduce_low(feat_low)
        mid = self.conv_reduce_mid(feat_mid)
        high = self.conv_reduce_high(feat_high)

        # 3. 高→中，上采样+融合
        high_up = F.interpolate(high, size=mid.shape[2:], mode='bilinear', align_corners=False)
        mid_fused = self.conv_fuse_mid(high_up + mid)

        # 4. 中→低，上采样+融合
        mid_up = F.interpolate(mid_fused, size=low.shape[2:], mode='bilinear', align_corners=False)
        low_fused = self.conv_fuse_low(mid_up + low)

        # 5. 分割预测
        seg_logits_small = self.seg_head(low_fused)
        seg_logits = F.interpolate(seg_logits_small, size=(H, W), mode='bilinear', align_corners=False)

        # 6. 分割概率用于姿态
        seg_probs_small = torch.sigmoid(seg_logits_small)

        # 7. 准备姿态特征：上采样到更高分辨率
        pose_feat = F.interpolate(low_fused, scale_factor=2.0, mode='bilinear', align_corners=False)
        seg_up = F.interpolate(seg_probs_small, size=pose_feat.shape[2:], mode='bilinear', align_corners=False)

        # 8. 融合分割概率与特征，并生成姿态特征
        combined = torch.cat([pose_feat, seg_up], dim=1)
        pose_feat_refined = self.seg_to_pose_conv(combined)

        # 9. 姿态预测
        pose_logits = self.pose_head(pose_feat_refined)

        if return_features:
            return seg_logits, pose_logits, low_fused, pose_feat_refined
        return seg_logits, pose_logits
