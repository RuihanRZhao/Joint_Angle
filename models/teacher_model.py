"""
teacher_model.py: Combines the segmentation and pose networks into an end-to-end teacher model.
The segmentation output is concatenated with the image and fed to the pose network.
"""

import torch
import torch.nn as nn
from models.segmentation_model import UNetSegmentation
from models.pose_model import PoseEstimationModel


class TeacherModel(nn.Module):
    def __init__(self, num_keypoints: int = 17, num_pafs: int = 32):
        super().__init__()
        # 分割分支，输出单通道人物语义分割 logits
        self.segmentation = UNetSegmentation(in_channels=3, num_classes=1)
        seg_out_channels = self.segmentation.conv_last.out_channels
        # 多人体姿态估计分支，输入通道 = RGB(3) + seg_logits(1)
        self.pose = PoseEstimationModel(in_channels=3 + seg_out_channels,
                                        num_keypoints=num_keypoints,
                                        num_pafs=num_pafs)

    def forward(self, x: torch.Tensor):
        # 语义分割
        seg_logits = self.segmentation(x)         # [B, 1, H, W]
        # 阻断梯度流向分割分支
        seg_for_pose = seg_logits.detach()
        # 拼接输入
        x_pose = torch.cat([x, seg_for_pose], dim=1)  # [B, 4, H, W]
        # 多人体姿态输出
        heatmaps, pafs = self.pose(x_pose)
        return seg_logits, heatmaps, pafs
