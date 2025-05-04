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
    """
    Multi-person keypoint MSE loss.
    pred_heatmaps: [B, K, H, W]
    keypoints_list: list of length B, each is tensor [P_i, K, 2] of normalized coords
    visibilities_list: list of length B, each is tensor [P_i, K] of visibility flags
    """
    def __init__(self, sigma=3):
        super(KeypointsLoss, self).__init__()
        self.sigma = sigma

    def forward(self, pred_heatmaps, keypoints_list, visibilities_list):
        total_loss = 0.0
        batch_size, K, H, W = pred_heatmaps.shape
        device = pred_heatmaps.device
        # build grid for gaussian computation
        grid_x, grid_y = torch.meshgrid(
            torch.arange(W, device=device),
            torch.arange(H, device=device),
            indexing='ij'
        )

        for i in range(batch_size):
            heatmap = pred_heatmaps[i]  # [K, H, W]
            kps_batch = keypoints_list[i]  # [P, K, 2]
            vis_batch = visibilities_list[i]  # [P, K]
            target = torch.zeros_like(heatmap)

            P = kps_batch.shape[0]
            for p in range(P):
                for k in range(K):
                    if vis_batch[p, k] > 0:
                        x_norm, y_norm = kps_batch[p, k]
                        x = int(x_norm * (W - 1))
                        y = int(y_norm * (H - 1))
                        if 0 <= x < W and 0 <= y < H:
                            dist = (grid_x - x) ** 2 + (grid_y - y) ** 2
                            target[k] += torch.exp(-dist / (2 * self.sigma ** 2))

            # compute MSE over all channels
            loss = F.mse_loss(heatmap, target, reduction='sum')
            total_vis = vis_batch.sum()
            total_loss += loss / (total_vis + 1e-6)

        return total_loss / batch_size


class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=2.0):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha  # Balance between distill and task loss
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.seg_loss = SegmentationLoss()
        self.keypoints_loss = KeypointsLoss()

    def forward(self, student_outputs, teacher_outputs, targets):
        # student_outputs: tuple(s_seg_logits, s_pose_logits)
        # teacher_outputs: tuple(t_seg_logits, t_pose_logits)
        # targets: tuple(mask, keypoints_list, visibilities_list)

        s_seg_logits, s_pose_logits = student_outputs
        t_seg_logits, t_pose_logits = teacher_outputs
        mask, keypoints_list, visibilities_list = targets

        # task loss on ground truth
        task_seg_loss = self.seg_loss(s_seg_logits, mask)
        task_pose_loss = self.keypoints_loss(s_pose_logits, keypoints_list, visibilities_list)
        task_loss = task_seg_loss + task_pose_loss

        # distillation loss on teacher soft labels
        s_seg_probs = F.log_softmax(s_seg_logits / self.temperature, dim=1)
        t_seg_probs = F.softmax(t_seg_logits / self.temperature, dim=1)
        seg_distill_loss = self.kl_div(s_seg_probs, t_seg_probs) * (self.temperature ** 2)

        # flatten pose heatmaps for KL
        s_pose_probs = F.log_softmax(
            s_pose_logits.view(s_pose_logits.size(0), -1) / self.temperature, dim=1
        )
        t_pose_probs = F.softmax(
            t_pose_logits.view(t_pose_logits.size(0), -1) / self.temperature, dim=1
        )
        pose_distill_loss = self.kl_div(s_pose_probs, t_pose_probs) * (self.temperature ** 2)

        distill_loss = seg_distill_loss + pose_distill_loss

        # total = alpha * distill + (1-alpha) * task
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * task_loss

        return total_loss, {
            'total_loss': total_loss.detach().item(),
            'task_seg_loss': task_seg_loss.detach().item(),
            'task_pose_loss': task_pose_loss.detach().item(),
            'seg_distill_loss': seg_distill_loss.detach().item(),
            'pose_distill_loss': pose_distill_loss.detach().item()
        }
