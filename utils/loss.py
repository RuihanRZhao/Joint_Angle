import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveMultiTaskLoss(nn.Module):
    """多任务自适应加权损失函数"""

    def __init__(self, num_tasks=2, init_log_var=0.0):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks) + init_log_var)

    def forward(self, task_losses):
        """
        输入:
            task_losses: 各任务损失列表 [L_seg, L_pose]
        输出:
            total_loss: 加权总损失
        """
        assert len(task_losses) == len(self.log_vars)

        total = 0.0
        for i, loss in enumerate(task_losses):
            precision = torch.exp(-self.log_vars[i])
            total += precision * loss + self.log_vars[i]

        return total


class SegmentationLoss(nn.Module):
    """分割任务复合损失函数"""

    def __init__(self, alpha=0.7, beta=0.3, gamma=2.0, smooth=1e-6):
        super().__init__()
        self.alpha = alpha  # Tversky损失参数
        self.beta = beta  # Tversky损失参数
        self.gamma = gamma  # Focal损失参数
        self.smooth = smooth  # 数值稳定性

    def tversky_loss(self, pred, target):
        # 将logits转换为概率
        pred = torch.sigmoid(pred)

        # 计算TP, FP, FN
        tp = torch.sum(pred * target)
        fp = torch.sum(pred * (1 - target))
        fn = torch.sum((1 - pred) * target)

        # Tversky指数
        numerator = tp + self.smooth
        denominator = tp + self.alpha * fp + self.beta * fn + self.smooth

        return 1 - (numerator / denominator)

    def focal_loss(self, pred, target):
        # 计算二元交叉熵
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        # Focal权重
        pt = torch.exp(-bce)
        focal_weight = (1 - pt) ** self.gamma

        return torch.mean(focal_weight * bce)

    def forward(self, pred, target):
        tversky = self.tversky_loss(pred, target)
        focal = self.focal_loss(pred, target)
        return 0.6 * tversky + 0.4 * focal


class PoseEstimationLoss(nn.Module):
    """姿态估计加权热图损失"""

    def __init__(self, beta=0.7, eps=1e-6):
        super().__init__()
        self.beta = beta  # 关键点区域权重系数
        self.eps = eps  # 防止除零

    def forward(self, pred, target):
        """
        输入:
            pred:   预测热图 [B, K, H, W]
            target: 目标热图 [B, K, H, W]
        输出:
            加权MSE损失
        """
        # 生成权重矩阵
        weights = self.beta * target + (1 - self.beta) * (1 - target)

        # 计算加权MSE
        loss = weights * (pred - target) ** 2

        # 按关键点数量归一化
        valid_keypoints = torch.sum(target.view(target.size(0), target.size(1), -1), dim=-1) > 0
        num_valid = torch.sum(valid_keypoints) + self.eps

        return torch.sum(loss) / num_valid


class Criterion(nn.Module):
    """总损失函数封装（支持热图+PAF双监督）"""

    def __init__(self):
        super().__init__()
        self.seg_loss = SegmentationLoss()
        self.hm_loss = PoseEstimationLoss()  # 热图损失
        self.paf_loss = PoseEstimationLoss(beta=0.5)  # PAF损失（调整参数）
        self.adaptive_loss = AdaptiveMultiTaskLoss(num_tasks=2)  # 保持2个任务

    def forward(self, seg_pred, seg_gt, pose_pred, hm_gt, paf_gt):
        """
        参数说明:
            pose_pred: 模型输出的姿态预测，应包含热图和PAF拼接
                       [B, 17+34, H, W] (假设17个关键点热图+34个PAF通道)
        """
        # 拆分预测结果
        hm_pred = pose_pred[:, :17]  # 热图预测
        paf_pred = pose_pred[:, 17:]  # PAF预测

        # 计算各项损失
        l_seg = self.seg_loss(seg_pred, seg_gt)
        l_hm = self.hm_loss(hm_pred, hm_gt)
        l_paf = self.paf_loss(paf_pred, paf_gt)

        # 合并姿态损失（热图和PAF视为同一任务）
        l_pose = 0.5 * l_hm + 0.5 * l_paf

        return self.adaptive_loss([l_seg, l_pose])

criterion = Criterion()