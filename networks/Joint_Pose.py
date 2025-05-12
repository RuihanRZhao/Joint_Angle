import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import InvertedResidual, GhostBottleneck

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import InvertedResidual, GhostBottleneck

class JointPoseNet(nn.Module):
    def __init__(self, num_joints=17):
        super(JointPoseNet, self).__init__()
        self.num_joints = num_joints
        # Backbone (MobileNetV2-like)
        self.backbone = nn.ModuleList()
        # 初始 conv: 3 -> 32
        self.backbone.append(nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        ))
        # InvertedResidual blocks
        mobilenet_cfg = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),  # 输出 24
            (6, 32, 3, 2),  # 输出 32
            (6, 64, 4, 2),  # 输出 64
            (6, 96, 3, 1),  # 输出 96
            (6, 160, 3, 1)  # 输出 160
        ]
        in_ch = 32
        for t, c, n, s in mobilenet_cfg:
            for i in range(n):
                stride = s if i == 0 else 1
                use_se = (c >= 64 and i == n-1)
                block = InvertedResidual(in_ch, c, stride=stride, expand_ratio=t, use_se=use_se)
                self.backbone.append(block)
                in_ch = c

        # High-res branch for early feature (after stage2 → 1/4 下采样, 通道 24)
        self.highres_branch = nn.ModuleList([
            GhostBottleneck(inp=24, mid_channels=48, oup=24, stride=1, use_se=False),
            GhostBottleneck(inp=24, mid_channels=48, oup=32, stride=1, use_se=True),
            GhostBottleneck(inp=32, mid_channels=64, oup=32, stride=1, use_se=False),
        ])

        # Upsampling low-res features to 1/4
        self.upsample1 = nn.Upsample(scale_factor=2)  # 1/16 → 1/8
        self.upsample2 = nn.Upsample(scale_factor=2)  # 1/8 → 1/4

        # Heatmap heads
        self.heatmap_conv1 = nn.Conv2d(160, num_joints, kernel_size=1)           # low-res → heatmap_init
        # refine_conv1 输入通道 = highres_branch 最终输出(32) + num_joints(17) = 49
        self.refine_conv1 = nn.Conv2d(32 + num_joints, 64, kernel_size=3, padding=1)
        self.heatmap_conv2 = nn.Conv2d(64, num_joints, kernel_size=1)           # refine → heatmap_refine

    def forward(self, x):
        features = x
        highres_input = None

        # 1. Backbone 前向传播，捕获第 2 阶段（idx==3）的特征
        for idx, layer in enumerate(self.backbone):
            features = layer(features)
            if idx == 3:
                highres_input = features  # 通道数 24, 分辨率 1/4

        # 2. 处理 low-res 特征
        #    features 此时通道 160，分辨率 1/16
        up_feat1 = self.upsample1(features)  # 1/8, ch=160
        up_feat2 = self.upsample2(up_feat1)  # 1/4, ch=160

        # 3. 第一阶段热图
        heatmap_init = torch.sigmoid(self.heatmap_conv1(up_feat2))  # [B, num_joints, H, W]

        # 4. highres_branch：对 early 特征做轻量化处理
        hr_feat = highres_input
        for block in self.highres_branch:
            hr_feat = block(hr_feat)  # 最终 hr_feat 通道数为 32, 分辨率 1/4

        # 5. 拼接高/低分辨率分支输出，送入 refine 卷积
        combined = torch.cat([hr_feat, heatmap_init], dim=1)  # ch = 32 + num_joints = 49
        refine_feat = self.refine_conv1(combined)
        heatmap_refine = torch.sigmoid(self.heatmap_conv2(refine_feat))  # [B, num_joints, H, W]

        # 6. DSNT 计算 keypoints 归一化坐标（[-1,1]）
        B, J, H, W = heatmap_refine.shape
        heatmap_flat = heatmap_refine.view(B, J, -1)
        prob = F.softmax(heatmap_flat, dim=2)

        grid_y = torch.linspace(-1.0, 1.0, steps=H, dtype=prob.dtype, device=prob.device)
        grid_x = torch.linspace(-1.0, 1.0, steps=W, dtype=prob.dtype, device=prob.device)
        grid_y = grid_y.unsqueeze(1).repeat(1, W).view(-1)
        grid_x = grid_x.repeat(H)

        grid_x = grid_x.unsqueeze(0).unsqueeze(0)  # [1,1,H*W]
        grid_y = grid_y.unsqueeze(0).unsqueeze(0)
        x_coords = torch.sum(prob * grid_x, dim=2)
        y_coords = torch.sum(prob * grid_y, dim=2)
        keypoints = torch.stack([x_coords, y_coords], dim=2)  # [B, J, 2]

        return heatmap_init, heatmap_refine, keypoints
