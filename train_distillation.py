# file: train_distillation.py

import math
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader

# Import the models (assuming they are in the project package)
from models.teacher_model import TeacherModel
from models.student_model import StudentModel
# from models.segmentation_model import SegmentationModel  # existing teacher sub-model class
# from models.pose_model import PoseModel                  # existing teacher sub-model class
# from data.dataset import PoseSegDataset                  # your dataset class providing images, mask GT, keypoint GT

def train_distillation(teacher_seg_weights: str, teacher_pose_weights: str,
                       data_dir: str, epochs: int = 50, batch_size: int = 16,
                       learning_rate: float = 1e-3, output_path: str = "distilled_model.pth"):
    # 1. Initialize teacher and student models
    # Load pre-trained teacher sub-models (assumes model classes and weights are available)
    # seg_model = SegmentationModel(...)
    # pose_model = PoseModel(...)
    # seg_model.load_state_dict(torch.load(teacher_seg_weights))
    # pose_model.load_state_dict(torch.load(teacher_pose_weights))
    # For simplicity, here we assume seg_model and pose_model are already constructed and loaded:
    seg_model = ...   # (Replace with actual segmentation model loading)
    pose_model = ...  # (Replace with actual pose model loading)

    teacher_model = TeacherModel(seg_model, pose_model)
    student_model = StudentModel(num_keypoints=pose_model.num_keypoints if hasattr(pose_model, 'num_keypoints') else <NUM_KEYPOINTS>,
                                 seg_channels=1, pretrained_backbone=True)
    student_model.train()  # set student in training mode

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)

    # 2. Prepare data
    # dataset = PoseSegDataset(data_dir, transform=..., target_transform=...)
    # train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_loader = DataLoader(...)  # (Use actual dataset loader)

    # 3. Define loss functions
    # Segmentation losses
    seg_criterion = nn.BCEWithLogitsLoss()  # for binary segmentation (if multi-class, use CrossEntropyLoss)
    # Pose losses
    pose_criterion = nn.MSELoss()  # MSE for keypoint heatmap regression
    # Optimizer
    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)

    # Distillation loss weights
    lambda_kd_seg = 1.0     # weight for segmentation soft distillation (KL)
    lambda_kd_pose = 1.0    # weight for pose distillation (heatmap L2)
    lambda_feat = 0.5       # weight for feature map distillation
    lambda_sup_seg = 1.0    # weight for supervised seg loss
    lambda_sup_pose = 1.0   # weight for supervised pose loss

    # Temperature for distillation (if using softmax + KL)
    T = 1.0  # (can be higher if softer distillation desired)

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            images, seg_mask_gt, pose_gt = batch  # images: [B,3,H,W], seg_mask_gt: [B,H,W] or [B,1,H,W], pose_gt: [B, num_kpts, Hh, Wh] or keypoint coords
            images = images.to(device)
            seg_mask_gt = seg_mask_gt.to(device)
            pose_gt = pose_gt.to(device)

            # Forward pass
            with torch.no_grad():
                # Get teacher outputs (and features)
                t_seg_logits, t_pose_out, t_seg_feat, t_pose_feat = teacher_model(images, return_features=True)
                # Detach teacher outputs so that they are treated as constants
                t_seg_logits = t_seg_logits.detach()
                t_pose_out = t_pose_out.detach()
                if t_seg_feat is not None:
                    t_seg_feat = t_seg_feat.detach()
                if t_pose_feat is not None:
                    t_pose_feat = t_pose_feat.detach()

            # Student forward (get outputs and its features for distillation)
            s_seg_logits, s_pose_out, s_feat_seg, s_feat_pose = student_model(images, return_features=True)

            # Ensure shapes are compatible (for example, if teacher outputs heatmaps at different resolution)
            # If needed, resize teacher pose output to student output resolution:
            if t_pose_out.shape != s_pose_out.shape:
                t_pose_out_resized = F.interpolate(t_pose_out, size=s_pose_out.shape[2:], mode='bilinear', align_corners=False)
            else:
                t_pose_out_resized = t_pose_out

            # 4. Compute losses
            # Supervised segmentation loss (ground truth vs student)
            if seg_mask_gt.ndim == 3:
                # shape [B,H,W], expand to [B,1,H,W] for BCE
                seg_mask_gt = seg_mask_gt.unsqueeze(1)
            loss_sup_seg = seg_criterion(s_seg_logits, seg_mask_gt.float())

            # Supervised pose loss (ground truth vs student)
            if s_pose_out.shape == pose_gt.shape:
                # If pose_gt is heatmap of same shape
                loss_sup_pose = pose_criterion(s_pose_out, pose_gt.float())
            else:
                # If pose_gt provided as coordinates, convert student heatmaps to coordinates and use L1 (Not common, but handle if needed)
                # (Here we assume heatmap ground truth; adapt accordingly if not)
                loss_sup_pose = pose_criterion(s_pose_out, pose_gt.float())

            # Knowledge distillation losses:
            # Segmentation KD (KL divergence between teacher & student segmentation distributions)
            if t_seg_logits.shape[1] > 1:
                # Multi-class segmentation: use softmax + KL
                # Compute soft targets (teacher probabilities) and student's log probabilities
                t_seg_prob = F.softmax(t_seg_logits / T, dim=1)
                s_seg_log_prob = F.log_softmax(s_seg_logits / T, dim=1)
                loss_kd_seg = F.kl_div(s_seg_log_prob, t_seg_prob, reduction='batchmean') * (T * T)
            else:
                # Binary segmentation: use teacher's probability as target in a soft binary cross-entropy
                t_seg_prob = torch.sigmoid(t_seg_logits / T)
                loss_kd_seg = F.binary_cross_entropy_with_logits(s_seg_logits, t_seg_prob, reduction='mean') * (T * T)

            # Pose KD (L2 loss between teacher and student heatmaps)
            loss_kd_pose = F.mse_loss(s_pose_out, t_pose_out_resized, reduction='mean')

            # Feature-based KD: L2 between intermediate feature maps
            loss_feat = 0.0
            if t_seg_feat is not None and s_feat_seg is not None:
                # If teacher seg feature resolution differs, upsample teacher feature to student feature size
                if t_seg_feat.shape[2:] != s_feat_seg.shape[2:]:
                    t_seg_feat_resized = F.interpolate(t_seg_feat, size=s_feat_seg.shape[2:], mode='bilinear', align_corners=False)
                else:
                    t_seg_feat_resized = t_seg_feat
                loss_feat += F.mse_loss(s_feat_seg, t_seg_feat_resized)
            if t_pose_feat is not None and s_feat_pose is not None:
                # Align teacher pose feature map size with student pose feature
                if t_pose_feat.shape[2:] != s_feat_pose.shape[2:]:
                    t_pose_feat_resized = F.interpolate(t_pose_feat, size=s_feat_pose.shape[2:], mode='bilinear', align_corners=False)
                else:
                    t_pose_feat_resized = t_pose_feat
                loss_feat += F.mse_loss(s_feat_pose, t_pose_feat_resized)
            # If teacher features are not available, feature loss will remain 0.

            # Total loss (weighted sum of all components)
            total_distill_loss = (lambda_sup_seg * loss_sup_seg +
                                   lambda_sup_pose * loss_sup_pose +
                                   lambda_kd_seg * loss_kd_seg +
                                   lambda_kd_pose * loss_kd_pose +
                                   lambda_feat * loss_feat)

            # 5. Optimize student model
            optimizer.zero_grad()
            total_distill_loss.backward()
            optimizer.step()

            total_loss += total_distill_loss.item()

        # Print average loss for the epoch (for monitoring)
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Distillation loss: {avg_loss:.4f}")

    # 6. Save the trained student model
    torch.save(student_model.state_dict(), output_path)
    print(f"Distilled model saved to {output_path}")
