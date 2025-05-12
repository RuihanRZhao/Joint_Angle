import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCCLoss(nn.Module):
    def __init__(self, x_weight=1.0, y_weight=1.0, smoothing=0.1):
        """
        SimCC 损失：使用 KLDivLoss 计算 soft logits 和 soft targets 的差异。
        支持 label smoothing。
        """
        super(SimCCLoss, self).__init__()
        self.x_weight = x_weight
        self.y_weight = y_weight
        self.smoothing = smoothing
        self.kl_loss = nn.KLDivLoss(reduction='none')  # 将 mask 应用于每个样本后再聚合

    def smooth_one_hot(self, target: torch.Tensor, classes: int, smoothing=0.1):
        """
        将 one-hot 向量转为 smoothed label。
        target: [N] long，表示类别索引。
        return: [N, classes] float
        """
        with torch.no_grad():
            target = target.view(-1)
            smooth = torch.full((target.size(0), classes), smoothing / (classes - 1), device=target.device)
            smooth.scatter_(1, target.unsqueeze(1), 1.0 - smoothing)
        return smooth

    def forward(self, pred_x, pred_y, target_x, target_y, mask):
        """
        pred_x/pred_y: [B, K, C]
        target_x/target_y: [B, K] long，表示类别索引（如 SimCC bin）
        mask: [B, K]
        """
        B, K, Cx = pred_x.shape
        _, _, Cy = pred_y.shape

        pred_x = pred_x.view(B * K, Cx)
        pred_y = pred_y.view(B * K, Cy)
        target_x = target_x.view(-1)  # [B*K]
        target_y = target_y.view(-1)
        mask = mask.view(-1)

        # 归一化 logits：log_softmax 作为 KLDivLoss 的输入
        log_prob_x = F.log_softmax(pred_x, dim=1)
        log_prob_y = F.log_softmax(pred_y, dim=1)

        # 生成 smoothed label
        soft_target_x = self.smooth_one_hot(target_x, Cx, self.smoothing)
        soft_target_y = self.smooth_one_hot(target_y, Cy, self.smoothing)

        # KL divergence：逐样本计算 loss
        loss_x = self.kl_loss(log_prob_x, soft_target_x).sum(dim=1)  # [B*K]
        loss_y = self.kl_loss(log_prob_y, soft_target_y).sum(dim=1)  # [B*K]

        # 只统计标注可用的 keypoint
        loss_x = loss_x * mask
        loss_y = loss_y * mask

        valid = mask.sum().clamp(min=1.0)
        mean_x = loss_x.sum() / valid
        mean_y = loss_y.sum() / valid

        total = self.x_weight * mean_x + self.y_weight * mean_y
        return {
            'total_loss': total,
            'x_loss': mean_x,
            'y_loss': mean_y
        }
