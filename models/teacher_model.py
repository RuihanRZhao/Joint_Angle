"""
teacher_model.py: Combines the segmentation and pose networks into an end-to-end teacher model.
The segmentation output is concatenated with the image and fed to the pose network.
Bottom-Up 解码逻辑已内嵌，无需外部依赖。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.segmentation_model import UNetSegmentation
from models.pose_model import PoseEstimationModel
import numpy as np
import cv2

# 内嵌 Bottom-Up 解码工具
def _find_peaks(heatmap, thresh=0.1):
    peaks = []
    h, w = heatmap.shape
    for y in range(1, h-1):
        for x in range(1, w-1):
            v = heatmap[y, x]
            if v > thresh and v >= heatmap[y-1, x] and v >= heatmap[y+1, x] \
                       and v >= heatmap[y, x-1] and v >= heatmap[y, x+1]:
                peaks.append((x, y, v))
    return peaks

# skeleton 对应的索引对（COCO 0-based）
_SKELETON = [
    (15,13),(13,11),(16,14),(14,12),(11,12),
    (5,11),(6,12),(5,6),(5,7),(7,9),
    (6,8),(8,10),(1,2),(0,1),(0,2),
    (1,3),(2,4),(3,5),(4,6)
]

class TeacherModel(nn.Module):
    def __init__(self, num_keypoints: int = 17, num_pafs: int = 32,
                 heatmap_thresh: float = 0.1, paf_score_thresh: float = 0.05):
        super().__init__()
        # 分割分支
        self.segmentation = UNetSegmentation(in_channels=3, num_classes=1)
        seg_out_channels = self.segmentation.conv_last.out_channels
        # 姿态分支
        self.pose = PoseEstimationModel(
            in_channels=3 + seg_out_channels,
            num_keypoints=num_keypoints,
            num_pafs=num_pafs
        )
        self.heatmap_thresh = heatmap_thresh
        self.paf_score_thresh = paf_score_thresh

    def forward(self, x: torch.Tensor):
        # 1) 语义分割
        seg_logits = self.segmentation(x)
        seg_for_pose = seg_logits.detach()

        # 2) 姿态预测
        x_pose = torch.cat([x, seg_for_pose], dim=1)
        heatmaps, pafs = self.pose(x_pose)  # [B,K,h,w], [B,2L,h,w]

        # 3) 内嵌 Bottom-Up 解码
        multi_kps = []
        B, K, h, w = heatmaps.shape
        for b in range(B):
            hm = heatmaps[b].detach().cpu().numpy()
            pf = pafs[b].detach().cpu().numpy()
            # 3.1 per-keypoint peaks
            all_peaks = [_find_peaks(hm[k], self.heatmap_thresh) for k in range(K)]
            # 3.2 简易将每个顶点独立成实例（示例级）
            persons = []
            for kpt_type, peaks in enumerate(all_peaks):
                for x0, y0, s0 in peaks:
                    coords = np.zeros((K,2), dtype=float)
                    coords[kpt_type] = [x0 / w * x.shape[3], y0 / h * x.shape[2]]
                    persons.append(coords)
            multi_kps.append(persons)

        return seg_logits, heatmaps, pafs, multi_kps