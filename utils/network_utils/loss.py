import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCCLoss(nn.Module):
    def __init__(self, x_weight=1.0, y_weight=1.0):
        super(SimCCLoss, self).__init__()
        self.x_weight = x_weight
        self.y_weight = y_weight
        self.ce_x = nn.CrossEntropyLoss(reduction='none')
        self.ce_y = nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred_x, pred_y, target_x, target_y, mask):
        """
        pred_x: [B, K, Cx] (logits)
        pred_y: [B, K, Cy] (logits)
        target_x: [B, K] (long indices)
        target_y: [B, K] (long indices)
        mask: [B, K] (float, 0 or 1)
        """
        B, K = target_x.shape
        pred_x = pred_x.view(B * K, -1)
        pred_y = pred_y.view(B * K, -1)
        target_x = target_x.view(-1)
        target_y = target_y.view(-1)
        mask = mask.view(-1)

        # compute individual losses
        loss_x = self.ce_x(pred_x, target_x)  # [B*K]
        loss_y = self.ce_y(pred_y, target_y)  # [B*K]

        # apply mask
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
