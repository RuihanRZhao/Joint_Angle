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
    def __init__(self, reduction='mean'):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, pred_heatmap, gt_heatmap):
        # 先做 sigmoid -> [B,K,H',W']
        pred_prob = torch.sigmoid(pred_heatmap)
        # 如果预测和 GT 分辨率不一样，重采样到 GT 大小
        if pred_prob.shape[2:] != gt_heatmap.shape[2:]:
            pred_prob = F.interpolate(
                pred_prob,
                size=gt_heatmap.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        return self.mse(pred_prob, gt_heatmap)


class PAFLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.l1 = nn.L1Loss(reduction=reduction)

    def forward(self, pred_paf, gt_paf):
        # 如果预测 PAF 尺寸与 GT 不匹配，先重采样
        if pred_paf.shape[2:] != gt_paf.shape[2:]:
            pred_paf = F.interpolate(
                pred_paf,
                size=gt_paf.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        return self.l1(pred_paf, gt_paf)


class MultiTaskLoss(nn.Module):
    """
    Combine segmentation, heatmap, and PAF losses.
    """

    def __init__(self,
                 weight_seg: float = 1.0,
                 weight_hm: float = 1.0,
                 weight_paf: float = 1.0,
                 pos_weight: torch.Tensor = None):
        super().__init__()
        self.seg_loss = SegmentationLoss(pos_weight=pos_weight)
        self.hm_loss = HeatmapLoss()
        self.paf_loss = PAFLoss()
        self.w_seg = weight_seg
        self.w_hm = weight_hm
        self.w_paf = weight_paf

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
        loss_hm = self.hm_loss(pred_hm, gt_hm)
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
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.lambda_dist = lambda_dist

    def forward(self, student_outputs, teacher_outputs, targets):
        """
        student_outputs: (s_seg, s_hm, s_paf)
        teacher_outputs: (t_seg, t_hm, t_paf) (already detached)
        targets: (gt_mask, gt_kps_list, gt_vis_list)
        """
        s_seg, s_hm, s_paf = student_outputs
        t_seg, t_hm, t_paf = teacher_outputs
        gt_mask, gt_kps, gt_vis = targets

        # —— 分割蒸馏 ——
        # 1) 老师的分割概率
        t_prob = torch.sigmoid(t_seg.detach())
        # 2) 对齐到 GT mask 的空间尺寸
        target_size = gt_mask.shape[2:]  # (H_gt, W_gt)
        if s_seg.shape[2:] != target_size:
            s_seg = F.interpolate(s_seg,
                                  size=target_size,
                                  mode='bilinear',
                                  align_corners=False)
        if t_prob.shape[2:] != target_size:
            t_prob = F.interpolate(t_prob,
                                   size=target_size,
                                   mode='bilinear',
                                   align_corners=False)
        # 3) 用 BCEWithLogits 计算蒸馏损失
        seg_loss = F.binary_cross_entropy_with_logits(s_seg, t_prob)

        # —— 热力图蒸馏 ——
        # 对齐后做 L2
        hm_pred = torch.sigmoid(s_hm)
        hm_tgt  = torch.sigmoid(t_hm.detach())
        if hm_pred.shape[2:] != hm_tgt.shape[2:]:
            hm_pred = F.interpolate(hm_pred,
                                    size=hm_tgt.shape[2:],
                                    mode='bilinear',
                                    align_corners=False)
        hm_loss = F.mse_loss(hm_pred, hm_tgt)

        # —— PAF 蒸馏 ——
        # 对齐后做 L1
        paf_pred = s_paf
        paf_tgt  = t_paf.detach()
        if paf_pred.shape[2:] != paf_tgt.shape[2:]:
            paf_pred = F.interpolate(paf_pred,
                                     size=paf_tgt.shape[2:],
                                     mode='bilinear',
                                     align_corners=False)
        paf_loss = F.l1_loss(paf_pred, paf_tgt)

        # —— 总损失加权 ——
        total_loss = self.alpha * seg_loss \
                   + self.beta  * hm_loss \
                   + self.gamma * paf_loss

        return total_loss, {
            'seg_loss'     : seg_loss,
            'hm_loss'      : hm_loss,
            'paf_loss'     : paf_loss,
            'distill_loss': total_loss
        }