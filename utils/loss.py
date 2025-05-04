import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
from torch import Tensor

class SegmentationLoss(nn.Module):
    """
    Binary segmentation loss using BCEWithLogits.
    """
    def __init__(self, pos_weight=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, pred_logits: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_logits: [B,1,H,W] raw logits
            gt_mask:     [B,1,H,W] binary mask (0 or 1)
        """
        return self.bce(pred_logits, gt_mask)


class HeatmapLoss(nn.Module):
    """
    Heatmap loss for keypoint confidence maps using MSE.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred_hm: torch.Tensor, gt_hm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_hm: [B,K,h,w] raw heatmaps logits
            gt_hm:   [B,K,h,w] target heatmaps (0-1 floats)
        """
        # apply sigmoid to logits
        pred_prob = torch.sigmoid(pred_hm)
        return self.mse(pred_prob, gt_hm)


class PAFLoss(nn.Module):
    """
    Part Affinity Field loss using MSE.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred_paf: torch.Tensor, gt_paf: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_paf: [B,2L,h,w] raw PAF logits
            gt_paf:   [B,2L,h,w] target PAF vectors
        """
        return self.mse(pred_paf, gt_paf)


class MultiTaskLoss(nn.Module):
    """
    Combine segmentation, heatmap, and PAF losses.
    """
    def __init__(self,
                 weight_seg: float = 1.0,
                 weight_hm:  float = 1.0,
                 weight_paf: float = 1.0,
                 pos_weight: torch.Tensor = None):
        super().__init__()
        self.seg_loss = SegmentationLoss(pos_weight=pos_weight)
        self.hm_loss  = HeatmapLoss()
        self.paf_loss = PAFLoss()
        self.w_seg    = weight_seg
        self.w_hm     = weight_hm
        self.w_paf    = weight_paf

    def forward(self,
                pred_seg: torch.Tensor,
                gt_mask: torch.Tensor,
                pred_hm: torch.Tensor,
                gt_hm: torch.Tensor,
                pred_paf: torch.Tensor,
                gt_paf: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_seg: logits [B,1,H,W]
            gt_mask:  [B,1,H,W]
            pred_hm:  [B,K,h,w]
            gt_hm:    [B,K,h,w]
            pred_paf: [B,2L,h,w]
            gt_paf:   [B,2L,h,w]

        Returns:
            total_loss, dict of individual losses
        """
        loss_seg = self.seg_loss(pred_seg, gt_mask)
        loss_hm  = self.hm_loss(pred_hm, gt_hm)
        loss_paf = self.paf_loss(pred_paf, gt_paf)
        total = self.w_seg * loss_seg + self.w_hm * loss_hm + self.w_paf * loss_paf
        return total, {"loss_seg": loss_seg, "loss_hm": loss_hm, "loss_paf": loss_paf}


class DistillationLoss(nn.Module):
    """
    蒸馏分割、Heatmap 和 PAF：
      - seg_task:   学生分割对 GT 的 BCE Loss
      - seg_dist:   学生分割对教师分割概率的 MSE Loss
      - hm_dist:    学生 heatmap 对教师 heatmap 概率的 MSE Loss
      - paf_dist:   学生 PAF 对教师 PAF 向量的 MSE Loss
      总 loss = seg_task + λ_dist*(seg_dist + hm_dist + paf_dist)
    """
    def __init__(self, lambda_dist: float = 1.0):
        super().__init__()
        self.bce   = nn.BCEWithLogitsLoss()
        self.mse   = nn.MSELoss()
        self.lambda_dist = lambda_dist

    def forward(self,
                student_outputs: Tuple[Tensor,Tensor,Tensor],
                teacher_outputs: Tuple[Tensor,Tensor,Tensor],
                ground_truth: Tuple[Tensor,Tensor,List[Tensor]]):
        # 拆包
        s_seg, s_hm, s_paf = student_outputs
        t_seg, t_hm, t_paf = teacher_outputs
        gt_mask, _, _      = ground_truth

        # 任务损失：分割对 GT
        loss_seg = self.bce(s_seg, gt_mask)

        # 蒸馏损失
        # 分割概率 MSE
        p_s = torch.sigmoid(s_seg)
        p_t = torch.sigmoid(t_seg.detach())
        loss_seg_dist = self.mse(p_s, p_t)

        # heatmap 概率 MSE
        hm_s = torch.sigmoid(s_hm)
        hm_t = torch.sigmoid(t_hm.detach())
        loss_hm_dist = self.mse(hm_s, hm_t)

        # PAF 直接 MSE
        loss_paf_dist = self.mse(s_paf, t_paf.detach())

        # 合计
        loss_distill = loss_seg_dist + loss_hm_dist + loss_paf_dist
        total_loss = loss_seg + self.lambda_dist * loss_distill

        # 输出 (loss, metrics)
        metrics = {
            "seg_loss":        loss_seg,
            "seg_distill":     loss_seg_dist,
            "hm_distill":      loss_hm_dist,
            "paf_distill":     loss_paf_dist,
            "distill_loss":    loss_distill
        }
        return total_loss, metrics
