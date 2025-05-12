import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import InvertedResidual, GhostBottleneck

class JointPoseNet(nn.Module):
    def __init__(self, num_joints=17):
        super(JointPoseNet, self).__init__()
        self.num_joints = num_joints
        # MobileNetV2 backbone settings
        self.backbone = nn.ModuleList()
        # Initial conv layer
        input_channels = 32
        self.backbone.append(nn.Sequential(
            nn.Conv2d(3, input_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU6(inplace=True)
        ))
        # Inverted residual blocks config (MobileNetV2 until output stride 16)
        mobilenet_cfg = [
            (1, 16, 1, 1),   # t, c, n, s (expand_ratio, out_channels, number of blocks, first_block_stride)
            (6, 24, 2, 2),   # output stride 4
            (6, 32, 3, 2),   # output stride 8
            (6, 64, 4, 2),   # output stride 16
            (6, 96, 3, 1),   # output stride 16 (no further downsample)
            (6, 160, 3, 1)   # output stride 16 (modified, originally stride 2 for 32 but we keep 16)
        ]
        in_ch = input_channels
        for t, c, n, s in mobilenet_cfg:
            for i in range(n):
                stride = s if i == 0 else 1
                use_se = False
                # Use SE in deeper layers for attention (e.g., last block of stages with output >=64 channels)
                if c >= 64 and i == n-1:
                    use_se = True
                block = InvertedResidual(in_ch, c, stride=stride, expand_ratio=t, use_se=use_se)
                self.backbone.append(block)
                in_ch = c
        backbone_out_channels = in_ch  # expected to be 160
        # High-resolution parallel branch (lightweight)
        # Takes feature from early backbone (after 1/4 resolution stage, 24 channels)
        self.highres_branch = nn.ModuleList([
            GhostBottleneck(inp=24, mid_channels=24 * 2, oup=24, stride=1, use_se=False),
            GhostBottleneck(inp=24, mid_channels=24 * 2, oup=32, stride=1, use_se=True),
            GhostBottleneck(inp=32, mid_channels=32 * 2, oup=32, stride=1, use_se=False),
        ])
        # Upsampling layers for low-res features (Dense Upsampling Convolution via PixelShuffle)
        self.upsample1 = nn.Upsample(scale_factor=2)   # upsample 1/16 -> 1/8
        self.upsample2 = nn.Upsample(scale_factor=2)   # upsample 1/8 -> 1/4
        # Convolution layers for heatmap prediction
        self.heatmap_conv1 = nn.Conv2d(160, num_joints, kernel_size=1)
        self.refine_conv1 = nn.Conv2d(64 + num_joints, 64, kernel_size=3, padding=1)
        self.heatmap_conv2 = nn.Conv2d(64, num_joints, kernel_size=1)


    def forward(self, x):
        features = x
        highres_input = None
        # Forward through backbone
        for idx, layer in enumerate(self.backbone):
            features = layer(features)
            # Capture feature after stage2 (index 3 in backbone list) for high-res branch input
            if idx == 3:
                highres_input = features

        # Low-res feature head (e.g., stage4 output)
        lr_feat = features  # final backbone output
        # High-res feature (upsampled stage2 output)
        hr_feat = highres_input

        # Upsample low-res features to 1/4 resolution
        up_feat1 = self.upsample1(features)  # 1/8, 64 channels
        up_feat2 = self.upsample2(up_feat1)  # 1/4, 64 channels
        # Stage 1 heatmaps
        heatmap_init = torch.sigmoid(self.heatmap_conv1(up_feat2))  # Sigmoid归一化热图到[0,1]范围 📎
        # 第2阶段细化：将高分辨率特征与第一阶段热图拼接
        combined = torch.cat([hr_feat, heatmap_init], dim=1)
        refine_feat = self.refine_conv1(combined)
        heatmap_refine = torch.sigmoid(self.heatmap_conv2(refine_feat))  # Sigmoid归一化热图到[0,1]范围 📎
        # DSNT for keypoints （修改后代码）
        B, J, H, W = heatmap_refine.shape
        # 展平热图并计算 softmax 概率分布
        heatmap_flat = heatmap_refine.view(B, J, -1)
        prob = F.softmax(heatmap_flat, dim=2)
        # 创建归一化坐标网格 [-1, 1]
        grid_y = torch.linspace(-1.0, 1.0, steps=H, dtype=prob.dtype, device=prob.device).unsqueeze(1).repeat(1,
                                                                                                              W).contiguous().view(
            -1)
        grid_x = torch.linspace(-1.0, 1.0, steps=W, dtype=prob.dtype, device=prob.device).repeat(H).contiguous()

        # 计算期望坐标
        grid_x = grid_x.unsqueeze(0).unsqueeze(0)  # [1,1,H*W]
        grid_y = grid_y.unsqueeze(0).unsqueeze(0)
        x_coords = torch.sum(prob * grid_x, dim=2)  # x坐标归一化值
        y_coords = torch.sum(prob * grid_y, dim=2)  # y坐标归一化值
        keypoints = torch.stack([x_coords, y_coords], dim=2)  # 拼接归一化坐标 ([-1,1]范围) 📎
        return heatmap_init, heatmap_refine, keypoints