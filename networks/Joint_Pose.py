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
        input_channels = 32
        self.backbone.append(nn.Sequential(
            nn.Conv2d(3, input_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU6(inplace=True)
        ))
        mobilenet_cfg = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 1)
        ]
        in_ch = input_channels
        for t, c, n, s in mobilenet_cfg:
            for i in range(n):
                stride = s if i == 0 else 1
                use_se = False
                if c >= 64 and i == n-1:
                    use_se = True
                block = InvertedResidual(in_ch, c, stride=stride, expand_ratio=t, use_se=use_se)
                self.backbone.append(block)
                in_ch = c
        backbone_out_channels = in_ch
        self.highres_branch = nn.ModuleList([  # High-resolution branch
            GhostBottleneck(inp=24, mid_channels=24 * 2, oup=24, stride=1, use_se=False),
            GhostBottleneck(inp=24, mid_channels=24 * 2, oup=32, stride=1, use_se=True),
            GhostBottleneck(inp=32, mid_channels=32 * 2, oup=32, stride=1, use_se=False),
        ])
        self.upsample1 = nn.Upsample(scale_factor=2)  # upsample 1/16 -> 1/8
        self.upsample2 = nn.Upsample(scale_factor=2)  # upsample 1/8 -> 1/4
        self.heatmap_conv1 = nn.Conv2d(160, num_joints, kernel_size=1)
        # Adjust input channels of refine_conv1 to match 58 (41 + num_joints)
        self.refine_conv1 = nn.Conv2d(41 + num_joints, 64, kernel_size=3, padding=1)  # Input channels: 41 + num_joints (58)
        self.heatmap_conv2 = nn.Conv2d(64, num_joints, kernel_size=1)

    def forward(self, x):
        features = x
        highres_input = None
        # Forward through backbone
        for idx, layer in enumerate(self.backbone):
            features = layer(features)
            if idx == 3:
                highres_input = features

        lr_feat = features  # final backbone output
        hr_feat = highres_input

        up_feat1 = self.upsample1(features)
        up_feat2 = self.upsample2(up_feat1)
        heatmap_init = torch.sigmoid(self.heatmap_conv1(up_feat2))  # Now works with 160 input channels
        combined = torch.cat([hr_feat, heatmap_init], dim=1)  # combined = [41 + 17] = 58 channels
        refine_feat = self.refine_conv1(combined)  # Input channels are now correctly set to 58
        heatmap_refine = torch.sigmoid(self.heatmap_conv2(refine_feat))

        B, J, H, W = heatmap_refine.shape
        heatmap_flat = heatmap_refine.view(B, J, -1)
        prob = F.softmax(heatmap_flat, dim=2)

        grid_y = torch.linspace(-1.0, 1.0, steps=H, dtype=prob.dtype, device=prob.device).unsqueeze(1).repeat(1, W).contiguous().view(-1)
        grid_x = torch.linspace(-1.0, 1.0, steps=W, dtype=prob.dtype, device=prob.device).repeat(H).contiguous()

        grid_x = grid_x.unsqueeze(0).unsqueeze(0)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0)
        x_coords = torch.sum(prob * grid_x, dim=2)
        y_coords = torch.sum(prob * grid_y, dim=2)
        keypoints = torch.stack([x_coords, y_coords], dim=2)
        return heatmap_init, heatmap_refine, keypoints
