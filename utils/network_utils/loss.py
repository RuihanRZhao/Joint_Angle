import torch
import torch.nn as nn

class HeatmapMSELoss(nn.Module):
    def __init__(self):
        super(HeatmapMSELoss, self).__init__()

    def forward(self, preds, targets, mask=None):
        if isinstance(preds, (tuple, list)):
            loss = 0.0
            for p in preds:
                loss += self._compute_loss(p, targets, mask)
            loss = loss / len(preds)
        else:
            loss = self._compute_loss(preds, targets, mask)
        return loss

    def _compute_loss(self, pred, target, mask):
        diff = pred - target
        if mask is not None:
            mask_expanded = mask[:, :, None, None].to(diff.device)
            diff = diff * mask_expanded
            num_present = mask_expanded.sum()
            if num_present.item() == 0:
                return torch.tensor(0.0, device=diff.device)
            loss = (diff ** 2).sum() / (num_present * pred.shape[-1] * pred.shape[-2])
        else:
            loss = (diff ** 2).mean()
        return loss

class PoseEstimationLoss(nn.Module):
    def __init__(self, coord_loss_weight=1.0):
        super(PoseEstimationLoss, self).__init__()
        self.heatmap_loss = HeatmapMSELoss()
        self.coord_weight = coord_loss_weight
        self.coord_loss_fn = nn.SmoothL1Loss(reduction='none')  # 改为逐点损失，便于应用mask

    def forward(self,
                heatmaps_preds,      # (heatmap_init, heatmap_refine)
                heatmaps_targets,    # [B, J, H, W]
                keypoints_preds,     # [B, J, 2], 归一化坐标
                keypoints_targets,   # [B, J, 2], 归一化坐标
                mask=None,           # [B, J]
                coord_weight=None):  # float 可选：当前 epoch 动态权重

        # 热图损失
        hm_loss = self.heatmap_loss(heatmaps_preds, heatmaps_targets, mask)

        # 坐标损失
        coord_loss = self.coord_loss_fn(keypoints_preds, keypoints_targets)  # [B, J, 2]

        if mask is not None:
            mask_exp = mask.unsqueeze(-1).to(coord_loss.device)  # [B, J, 1]
            coord_loss = coord_loss * mask_exp  # 屏蔽不可见关键点
            num_present = mask.sum()
            if num_present.item() == 0:
                coord_loss = torch.tensor(0.0, device=coord_loss.device)
            else:
                coord_loss = coord_loss.sum() / (num_present * 2)  # 平均每个点x,y
        else:
            coord_loss = coord_loss.mean()

        # 坐标损失权重（支持warmup）
        gamma = coord_weight if coord_weight is not None else 1.0
        total_loss = hm_loss + gamma * self.coord_weight * coord_loss

        return total_loss, {
            'loss_heatmap': hm_loss.item(),
            'loss_coord': coord_loss.item() if isinstance(coord_loss, torch.Tensor) else coord_loss,
            'coord_weight': gamma * self.coord_weight
        }