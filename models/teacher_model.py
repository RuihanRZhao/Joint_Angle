import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.segmentation_model import SegFormerSegmentation
from models.pose_model import ViTPoseModel

# Bottom-Up 解码工具：检测热图峰值
def _find_peaks(heatmap, thresh=0.1):
    peaks = []
    h, w = heatmap.shape
    for y in range(1, h-1):
        for x in range(1, w-1):
            v = heatmap[y, x]
            if v > thresh and \
               v >= heatmap[y-1, x] and v >= heatmap[y+1, x] and \
               v >= heatmap[y, x-1] and v >= heatmap[y, x+1]:
                peaks.append((x, y, v))
    return peaks

# COCO Skeleton 连接索引对 (0-based)
_SKELETON = [
    (15,13),(13,11),(16,14),(14,12),(11,12),
    (5,11),(6,12),(5,6),(5,7),(7,9),
    (6,8),(8,10),(1,2),(0,1),(0,2),
    (1,3),(2,4),(3,5),(4,6)
]

class TeacherModel(nn.Module):
    """
    Teacher model combining a pre-trained SegFormer segmentation and ViTPose estimation.
    Outputs segmentation logits, heatmaps, PAFs, and multi-person keypoints (multi_kps).
    """
    def __init__(
        self,
        seg_pretrained: str = "nvidia/segformer-b3-finetuned-ade-512-512",
        pose_backbone: str = "vit_base_patch16_224",
        pose_pretrained: bool = True,
        img_size: int = 480,
        num_keypoints: int = 17,
        num_pafs: int = 19,
        heatmap_thresh: float = 0.1,
        paf_score_thresh: float = 0.05
    ):
        super().__init__()
        # 分割分支
        self.segmentation = SegFormerSegmentation(
            pretrained_model_name=seg_pretrained,
            num_classes=1
        )
        # 姿态分支
        self.pose = ViTPoseModel(
            backbone_name=pose_backbone,
            pretrained=pose_pretrained,
            img_size=img_size,
            num_keypoints=num_keypoints,
            num_pafs=num_pafs
        )
        self.heatmap_thresh = heatmap_thresh
        self.paf_score_thresh = paf_score_thresh

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: RGB input tensor [B,3,H,W]
        Returns:
            seg_logits: [B,1,H,W]
            heatmaps:   [B,K,H/4,W/4]
            pafs:       [B,2*L,H/4,W/4]
            multi_kps:  List[B] of lists of keypoint coords per person
        """
        # 1) 语义分割
        seg_logits = self.segmentation(x)          # [B,1,h_small,w_small]
        seg_mask   = torch.sigmoid(seg_logits).detach()
        
        # ★ 新增：上采样到原图尺寸 ★
        seg_mask_up = F.interpolate(
            seg_mask, size=(x.shape[2], x.shape[3]),
            mode='bilinear', align_corners=False
        )
        
        # 拼接
        x_pose = torch.cat([x, seg_mask_up], dim=1)   # [B,4,H,W]
        heatmaps, pafs = self.pose(x_pose)         # [B,K,h,w], [B,2L,h,w]

        # 3) Bottom-Up 解码生成多人体关键点集合
        multi_kps = []
        B, K, h, w = heatmaps.shape
        # 遍历 batch
        for b in range(B):
            hm = heatmaps[b].detach().cpu().numpy()
            pf = pafs[b].detach().cpu().numpy()
            all_peaks = [_find_peaks(hm[k], self.heatmap_thresh) for k in range(K)]
            persons = []
            # 简易按类型分组，每个峰值视为单人实例
            for kpt_type, peaks in enumerate(all_peaks):
                for x0, y0, score in peaks:
                    coords = np.zeros((K,2), dtype=float)
                    # 缩放回原图坐标
                    coords[kpt_type] = [x0 / w * x.shape[3], y0 / h * x.shape[2]]
                    persons.append(coords)
            multi_kps.append(persons)

        return seg_logits, heatmaps, pafs, multi_kps