import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveMultiTaskLoss(nn.Module):
    """多任务自适应加权损失函数"""

    def __init__(self, num_tasks=2, init_log_var=0.0):
        super().__init__()
        # 日志方差参数，用于自适应任务权重
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
    """分割任务复合损失函数: Tversky + Focal"""

    def __init__(self, alpha=0.7, beta=0.3, gamma=2.0, smooth=1e-6):
        super().__init__()
        self.alpha = alpha  # Tversky FP 权重
        self.beta = beta    # Tversky FN 权重
        self.gamma = gamma  # Focal gamma 参数
        self.smooth = smooth  # 平滑项

    def tversky_loss(self, pred, target):
        pred = torch.sigmoid(pred)
        # 计算 TP, FP, FN
        tp = torch.sum(pred * target)
        fp = torch.sum(pred * (1 - target))
        fn = torch.sum((1 - pred) * target)
        # Tversky 指数
        numerator = tp + self.smooth
        denominator = tp + self.alpha * fp + self.beta * fn + self.smooth
        return 1 - (numerator / denominator)

    def focal_loss(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal_weight = (1 - pt) ** self.gamma
        return torch.mean(focal_weight * bce)

    def forward(self, pred, target):
        # 对齐 target 至 pred 大小
        if pred.shape[2:] != target.shape[2:]:
            target = F.interpolate(target, size=pred.shape[2:], mode='nearest')
        tversky = self.tversky_loss(pred, target)
        focal = self.focal_loss(pred, target)
        # 按比例加权
        return 0.6 * tversky + 0.4 * focal


class PoseEstimationLoss(nn.Module):
    """姿态估计加权热图 & PAF 损失"""

    def __init__(self, beta=0.7, eps=1e-6):
        super().__init__()
        self.beta = beta  # 关键点区域权重
        self.eps = eps    # 防除零

    def forward(self, pred, target):
        # 如果分辨率不同，上采样至 pred 大小
        if pred.shape[2:] != target.shape[2:]:
            target = F.interpolate(target, size=pred.shape[2:], mode='bilinear', align_corners=False)
        # 权重矩阵
        weights = self.beta * target + (1 - self.beta) * (1 - target)
        loss = weights * (pred - target) ** 2
        # 只对有目标的关键点归一化
        valid = torch.sum(target.view(target.size(0), target.size(1), -1), dim=-1) > 0
        num_valid = torch.sum(valid) + self.eps
        return torch.sum(loss) / num_valid


class Criterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.seg_loss   = SegmentationLoss()
        self.hm_loss    = PoseEstimationLoss()           # 热图部分
        self.paf_loss   = PoseEstimationLoss(beta=0.5)   # PAF 部分
        self.adaptive_loss = AdaptiveMultiTaskLoss(num_tasks=2)

    def forward(self, seg_pred, seg_gt, pose_pred, hm_gt, paf_gt=None):
        # 1) 分割损失
        l_seg = self.seg_loss(seg_pred, seg_gt)

        # 2) 动态拆分热图和 PAF
        num_kps  = hm_gt.size(1)      # 从 GT heatmap 读取通道数（一般 17）
        channels = pose_pred.size(1)  # 模型输出通道数

        # 切出热图部分
        hm_pred = pose_pred[:, :num_kps]
        l_hm    = self.hm_loss(hm_pred, hm_gt)

        # 切出 PAF 部分（仅当通道足够且提供了 paf_gt）
        if channels > num_kps and paf_gt is not None:
            paf_pred = pose_pred[:, num_kps:]
            l_paf    = self.paf_loss(paf_pred, paf_gt)
        else:
            l_paf = 0.0

        # 3) 将热图和 PAF 合并为一个姿态任务损失
        l_pose = 0.5 * l_hm + 0.5 * l_paf

        # 4) 自适应多任务加权
        total_loss = self.adaptive_loss([l_seg, l_pose])
        return total_loss


# 全局 criterion 实例
criterion = Criterion()
