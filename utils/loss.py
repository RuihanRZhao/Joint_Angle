import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationLoss(nn.Module):
    def __init__(self):
        super(SegmentationLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        # 如果 target 只有 H×W，没有通道维度，就加上一个单通道维度
        # pred: [N, C, H, W], target: [N, H, W] 或 [N, 1, H, W]
        if target.dim() == pred.dim() - 1:
            target = target.unsqueeze(1)
        return self.bce_loss(pred, target)


class KeypointsLoss(nn.Module):
    def forward(self, pred_heatmaps, keypoints_list, visibilities_list):
        """
        pred_heatmaps: [B, K, H, W]
        keypoints_list: list of [N_kp, 2] tensors
        visibilities_list: list of [N_kp] tensors
        """
        total_loss = 0
        batch_size = pred_heatmaps.size(0)

        for i in range(batch_size):
            kps = keypoints_list[i]  # [N_kp, 2]
            vis = visibilities_list[i]  # [N_kp]
            heatmap = pred_heatmaps[i]  # [K, H, W]

            # 生成目标热图
            target = torch.zeros_like(heatmap)
            for j, (x, y) in enumerate(kps):
                if vis[j] > 0:
                    x = int(x * (heatmap.size(2) - 1))
                    y = int(y * (heatmap.size(1) - 1))
                    if 0 <= x < heatmap.size(2) and 0 <= y < heatmap.size(1):
                        # 高斯分布
                        grid_x, grid_y = torch.meshgrid(
                            torch.arange(heatmap.size(2)),
                            torch.arange(heatmap.size(1)),
                            indexing='ij'
                        )
                        dist = (grid_x - x) ** 2 + (grid_y - y) ** 2
                        target[j] = torch.exp(-dist / (2 * 3 ** 2))

            # 计算MSE
            loss = F.mse_loss(heatmap, target, reduction='sum')
            total_loss += loss / (vis.sum() + 1e-6)

        return total_loss / batch_size


class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=2.0):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha  # 平衡蒸馏损失和任务损失的权重
        self.temperature = temperature  # 温度参数
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.seg_loss = SegmentationLoss()
        self.keypoints_loss = KeypointsLoss()

    def forward(self, student_outputs, teacher_outputs, targets):
        """
        知识蒸馏损失
        student_outputs: 学生模型输出 (seg_logits, pose_logits)
        teacher_outputs: 教师模型输出 (seg_logits, pose_logits)
        targets: 真实标签 (mask, keypoints, visibilities)
        """
        s_seg_logits, s_pose_logits = student_outputs
        t_seg_logits, t_pose_logits = teacher_outputs
        mask, keypoints, visibilities = targets

        # 任务损失（针对真实标签）
        task_seg_loss = self.seg_loss(s_seg_logits, mask)
        task_pose_loss = self.keypoints_loss(s_pose_logits, keypoints, visibilities)
        task_loss = task_seg_loss + task_pose_loss

        # 蒸馏损失（针对教师模型的输出）
        # 分割蒸馏损失
        s_seg_probs = F.log_softmax(s_seg_logits / self.temperature, dim=1)
        t_seg_probs = F.softmax(t_seg_logits / self.temperature, dim=1)
        seg_distill_loss = self.kl_div(s_seg_probs, t_seg_probs) * (self.temperature ** 2)

        # 姿态估计蒸馏损失
        s_pose_probs = F.log_softmax(s_pose_logits.view(s_pose_logits.size(0), -1) / self.temperature, dim=1)
        t_pose_probs = F.softmax(t_pose_logits.view(t_pose_logits.size(0), -1) / self.temperature, dim=1)
        pose_distill_loss = self.kl_div(s_pose_probs, t_pose_probs) * (self.temperature ** 2)

        distill_loss = seg_distill_loss + pose_distill_loss

        # 总损失 = alpha * 蒸馏损失 + (1 - alpha) * 任务损失
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * task_loss

        return total_loss, {
            'total_loss': total_loss.item(),
            'task_seg_loss': task_seg_loss.item(),
            'task_pose_loss': task_pose_loss.item(),
            'seg_distill_loss': seg_distill_loss.item(),
            'pose_distill_loss': pose_distill_loss.item()
        }
