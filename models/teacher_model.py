"""
teacher_model.py: Combines the segmentation and pose networks into an end-to-end teacher model.
The segmentation output is concatenated with the image and fed to the pose network.
"""

import torch
import torch.nn as nn
from models.segmentation_model import UNetSegmentation
from models.pose_model import PoseEstimationModel


class TeacherModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 分割分支
        self.segmentation = UNetSegmentation()
        # 获取分割输出的通道数（num_classes）
        seg_out_channels = self.segmentation.conv_last.out_channels
        # 姿态分支：输入 = RGB 图 + 分割 logits
        self.pose = PoseEstimationModel(in_channels = 3 + seg_out_channels)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Forward pass: compute segmentation and pose.
        Returns a tuple (segmentation_logits, pose_heatmaps).
        """
        seg_logits = self.segmentation(x)  # shape: [B, C, H, W]
        # Convert seg_logits to probabilities (optional softmax) or use raw logits.
        # Here we use raw logits and allow pose network to learn from them.
        # Concatenate image and segmentation along channel dimension
        # If seg_logits has same HxW as x; ensure sizes match
        # If needed, upsample seg_logits to x.size() (assumed already same here).
        # Detach segmentation output so pose gradients don't flow back into seg branch during forward.
        seg_for_pose = seg_logits.detach()
        x_pose = torch.cat([x, seg_for_pose], dim=1)  # new channels = 3 + seg_channels
        pose_heatmaps = self.pose(x_pose)
        return seg_logits, pose_heatmaps
