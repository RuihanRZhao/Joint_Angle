import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCCLoss(nn.Module):
    def __init__(self, x_weight=1.0, y_weight=1.0, use_soft=True, reduction='mean'):
        """
        参数:
            x_weight/y_weight: x轴和y轴的损失权重
            use_soft: 是否启用soft label (默认True)
            reduction: 'mean' or 'none'
        """
        super(SimCCLoss, self).__init__()
        self.x_weight = x_weight
        self.y_weight = y_weight
        self.use_soft = use_soft
        self.reduction = reduction

        if self.use_soft:
            self.criterion_x = nn.KLDivLoss(reduction='none', log_target=False)
            self.criterion_y = nn.KLDivLoss(reduction='none', log_target=False)
        else:
            self.criterion_x = nn.CrossEntropyLoss(reduction='none')
            self.criterion_y = nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred_x, pred_y, target_x, target_y, mask):
        """
        输入:
            pred_x: [B, K, Cx] - raw logits
            pred_y: [B, K, Cy]
            target_x: [B, K, Cx] if soft else [B, K] indices
            target_y: [B, K, Cy] if soft else [B, K]
            mask: [B, K] - float mask (0 or 1)
        输出:
            loss dict: {'total_loss', 'x_loss', 'y_loss'}
        """
        B, K, _ = pred_x.shape
        mask = mask.view(-1)  # [B*K]

        # Reshape: [BK, C]
        pred_x = pred_x.view(B * K, -1)
        pred_y = pred_y.view(B * K, -1)

        if self.use_soft:
            # KLDiv expects log-probs
            log_pred_x = F.log_softmax(pred_x, dim=1)
            log_pred_y = F.log_softmax(pred_y, dim=1)
            target_x = target_x.view(B * K, -1)
            target_y = target_y.view(B * K, -1)

            loss_x = self.criterion_x(log_pred_x, target_x).sum(dim=1)  # [BK]
            loss_y = self.criterion_y(log_pred_y, target_y).sum(dim=1)
        else:
            # CrossEntropy expects class index
            target_x = target_x.view(B * K)
            target_y = target_y.view(B * K)
            loss_x = self.criterion_x(pred_x, target_x)  # [BK]
            loss_y = self.criterion_y(pred_y, target_y)

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
