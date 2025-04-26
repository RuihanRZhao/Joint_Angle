# file: models/teacher_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class TeacherModel(nn.Module):
    """
    End-to-end teacher model that sequentially applies segmentation and pose estimation.
    """
    def __init__(self, seg_model: nn.Module, pose_model: nn.Module):
        super(TeacherModel, self).__init__()
        self.seg_model = seg_model      # pre-trained segmentation model (e.g., human mask)
        self.pose_model = pose_model    # pre-trained pose estimation model

        # Ensure models are in evaluation mode (teacher is fixed during distillation)
        self.seg_model.eval()
        self.pose_model.eval()
        for p in self.seg_model.parameters():
            p.requires_grad = False
        for p in self.pose_model.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        Forward pass: get segmentation, then pose.
        If return_features is True, also return intermediate feature maps for distillation.
        """
        # Stage 1: Segmentation prediction (human mask)
        seg_logits = self.seg_model(x)            # shape [B, C_seg, H, W], C_seg usually 1 or 2 for binary mask
        # Convert segmentation logits to a mask for pose input
        if seg_logits.shape[1] == 1:
            # Binary mask segmentation (foreground vs background)
            seg_mask = torch.sigmoid(seg_logits)  # probability mask in [0,1]
        else:
            # Multi-class segmentation (assume index 0=background, 1=person)
            seg_soft = F.softmax(seg_logits, dim=1)  # per-pixel class probabilities
            # Use the "person" class probability as mask
            seg_mask = seg_soft[:, 1:2, ...]

        # Create a masked image by zeroing out background (broadcast mask to 3 channels if needed)
        if seg_mask.shape[1] == 1 and x.shape[1] == 3:
            seg_mask = seg_mask.repeat(1, 3, 1, 1)
        masked_image = x * seg_mask

        # Stage 2: Pose prediction (keypoint heatmaps or coordinates)
        pose_output = self.pose_model(masked_image)

        if not return_features:
            # Return only final outputs
            return seg_logits, pose_output

        # If requested, gather intermediate feature maps for distillation
        seg_feat = None
        pose_feat = None
        # Try to obtain features from internal layers if supported by models
        if hasattr(self.seg_model, "get_features"):
            # e.g., a method to get high-level feature map from segmentation model
            seg_feat = self.seg_model.get_features(x)
        if hasattr(self.pose_model, "get_features"):
            # e.g., a method to get the last feature map before keypoint output
            pose_feat = self.pose_model.get_features(masked_image)
        return seg_logits, pose_output, seg_feat, pose_feat
