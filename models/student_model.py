# optimized_seg_pose_mbv2.py
# Using MobileNetV2 backbone for lightweight segmentation + Bottom-Up pose decoder
# Shared backbone for efficiency, integrated multi-person Bottom-Up decoding in StudentModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import numpy as np
import cv2

# —— Bottom-Up 解码工具 ——
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

_SKELETON = [
    (15,13),(13,11),(16,14),(14,12),(11,12),
    (5,11),(6,12),(5,6),(5,7),(7,9),
    (6,8),(8,10),(1,2),(0,1),(0,2),
    (1,3),(2,4),(3,5),(4,6)
]

class MobileNetV2Seg(nn.Module):
    """
    MobileNetV2-based Segmentation
    """
    def __init__(self, num_classes=1, width_mult=1.0, pretrained_backbone=True):
        super().__init__()
        mb2 = mobilenet_v2(
            weights=MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained_backbone else None,
            width_mult=width_mult
        )
        self.features = mb2.features
        in_ch = mb2.last_channel
        self.decoder = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 2, num_classes, 1)
        )

    def forward(self, x):
        feats = self.features(x)
        up = F.interpolate(feats, scale_factor=32, mode='bilinear', align_corners=False)
        return self.decoder(up)

class StudentModel(nn.Module):
    """
    Shared Backbone Multi-Task Model
    Segmentation + Bottom-Up Pose (Heatmap+PAF) integrated decoder
    """
    def __init__(
        self,
        seg_width_mult=1.0,
        heatmap_thresh=0.1,
        paf_score_thresh=0.05
    ):
        super().__init__()
        # Shared MobileNetV2 backbone
        mb2 = mobilenet_v2(
            weights=MobileNet_V2_Weights.IMAGENET1K_V1,
            width_mult=seg_width_mult
        )
        self.features = mb2.features
        feat_ch = mb2.last_channel
        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(feat_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch // 2, 1, 1)
        )
        # Pose head: Heatmap + PAF light
        self.pose_hm = nn.Conv2d(feat_ch, len(_SKELETON)+2, kernel_size=1)  # 17 keypoints
        self.pose_paf = nn.Conv2d(feat_ch, len(_SKELETON)*2, kernel_size=1) # PAF channels
        self.heatmap_thresh = heatmap_thresh
        self.paf_score_thresh = paf_score_thresh

    def forward(self, x):
        feats = self.features(x)  # [B,C,h/32,w/32]
        # Segmentation
        seg_up = F.interpolate(feats, scale_factor=32, mode='bilinear', align_corners=False)
        seg_logits = self.seg_head(seg_up)  # [B,1,H,W]
        # Pose maps
        hm_map = self.pose_hm(feats)        # [B,17,h,w]
        paf_map = self.pose_paf(feats)      # [B,2L,h,w]

        # Bottom-Up 多人解码
        multi_kps = []
        B, K, h, w = hm_map.shape
        for b in range(B):
            hm = hm_map[b].detach().cpu().numpy()
            pf = paf_map[b].detach().cpu().numpy()
            all_peaks = [_find_peaks(hm[k], self.heatmap_thresh) for k in range(K)]
            persons = []
            for kpt_type, peaks in enumerate(all_peaks):
                for x0, y0, v0 in peaks:
                    coords = np.zeros((K, 2), dtype=float)
                    coords[kpt_type] = [x0 / w * x.shape[3], y0 / h * x.shape[2]]
                    persons.append(coords)
            multi_kps.append(persons)

        return seg_logits, hm_map, paf_map, multi_kps
