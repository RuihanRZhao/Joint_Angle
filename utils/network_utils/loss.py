import torch
import torch.nn as nn

class HeatmapMSELoss(nn.Module):
    """
    姿态估计热图回归损失:
    使用均方误差(MSE)计算预测热图与目标热图之间的损失。
    支持对未标注的关键点通道进行掩码(mask)，使其不计入损失。
    """
    def __init__(self):
        super(HeatmapMSELoss, self).__init__()

    def forward(self, preds, targets, mask=None):
        """
        参数:
        - preds: 模型输出的热图 (tensor，形状 [B, num_joints, H, W] 或包含多个阶段输出的tuple/list)
        - targets: 对应的目标热图 (tensor，同形状)
        - mask: 关键点掩码 (tensor，[B, num_joints])，若某关键点未标注则该位置为0。
        返回:
        - MSE损失值 (tensor)
        """
        # 如果模型有多阶段输出 (如返回heatmap1和heatmap2)，对每个阶段都计算损失
        if isinstance(preds, (tuple, list)):
            loss = 0.0
            for p in preds:
                loss += self._compute_loss(p, targets, mask)
            loss = loss / len(preds)  # 多阶段损失平均
        else:
            loss = self._compute_loss(preds, targets, mask)
        return loss

    def _compute_loss(self, pred, target, mask):
        # 计算单阶段输出的 MSELoss (考虑掩码)
        diff = pred - target  # 差值
        if mask is not None:
            # 将 mask 扩展为与热图相同形状，用于将未标注关节的误差置零
            # mask shape: [B, num_joints] -> [B, num_joints, 1, 1]
            mask_expanded = mask[:, :, None, None].to(diff.device)
            diff = diff * mask_expanded  # 未标注关键点对应diff为0，不产生损失
            # 仅对标注的关键点计算平均误差:
            num_present = mask_expanded.sum()  # 有效关键点总数量 (每个关键点的H*W像素都计入)
            if num_present.item() == 0:
                return torch.tensor(0.0, device=diff.device)
            # 将总MSE误差除以有效像素数 (每个有效关键点的热图像素均计入)
            loss = (diff ** 2).sum() / (num_present * pred.shape[-1] * pred.shape[-2])
        else:
            # 若无mask，直接计算全图平均MSE
            loss = (diff ** 2).mean()
        return loss

class PoseEstimationLoss(nn.Module):
    """
    综合热图回归与坐标回归的多任务损失:
    - 热图阶段: heatmap_init & heatmap_refine 使用 MSELoss
    - 坐标阶段: 直接预测 keypoints 使用 SmoothL1Loss
    - 使用线性 warmup 调度坐标损失权重
    """
    def __init__(self, coord_loss_weight=1.0, warmup_epochs=15):
        """
        coord_loss_weight: 最终坐标损失的权重
        warmup_epochs: 坐标损失权重 warmup 的 epoch 数
        """
        super(PoseEstimationLoss, self).__init__()
        self.heatmap_loss = HeatmapMSELoss()
        self.coord_loss = nn.SmoothL1Loss()
        self.coord_weight = coord_loss_weight
        self.warmup_epochs = warmup_epochs

    def forward(self,
                heatmaps_preds,   # tuple: (heatmap_init, heatmap_refine)
                heatmaps_targets, # tensor: [B, J, H, W]
                keypoints_preds,  # tensor: [B, J, 2]
                keypoints_targets,# tensor: [B, J, 2]
                mask=None,        # tensor: [B, J]
                current_epoch=None):
        # 1. 计算热图损失
        hm_loss = self.heatmap_loss(heatmaps_preds, heatmaps_targets, mask)

        # 2. 计算坐标回归损失
        coord_loss = self.coord_loss(keypoints_preds, keypoints_targets)

        # 3. 动态计算坐标损失权重 (linear warmup)
        if current_epoch is not None and self.warmup_epochs > 0:
            gamma = min(current_epoch / float(self.warmup_epochs), 1.0)
        else:
            gamma = 1.0

        # 4. 加权求和
        total_loss = hm_loss + gamma * self.coord_weight * coord_loss

        return total_loss, {
            'loss_heatmap': hm_loss.item(),
            'loss_coord': coord_loss.item(),
            'coord_weight': gamma * self.coord_weight
        }