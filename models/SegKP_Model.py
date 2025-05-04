import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

from utils.Evaluator import PoseEvaluator


class CoordinateAttention(nn.Module):
    """坐标注意力机制，替代原SE模块"""

    def __init__(self, in_channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mid_channels = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mid_channels, in_channels, 1)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, 1)

    def forward(self, x):
        identity = x
        B, C, H, W = x.size()

        # 水平池化
        x_h = self.pool_h(x)  # [B,C,H,1]
        # 垂直池化
        x_w = self.pool_w(x)  # [B,C,1,W]

        # 特征融合
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 分解注意力图
        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # 生成注意力权重
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        return identity * a_w * a_h


class DynamicConv2d(nn.Module):
    """动态卷积模块，用于替换部分3x3卷积"""

    def __init__(self, in_ch, out_ch, kernel_size=3, num_experts=4):
        super().__init__()
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, num_experts, 1),
            nn.Softmax(dim=1)
        )
        self.convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, kernel_size,
                      padding=kernel_size // 2, bias=False)
            for _ in range(num_experts)
        ])

    def forward(self, x):
        B, C, H, W = x.size()
        attn = self.router(x)  # [B, K, 1, 1]

        outputs = []
        for conv in self.convs:
            outputs.append(conv(x).unsqueeze(1))  # [B,1,C,H,W]

        outputs = torch.cat(outputs, dim=1)  # [B,K,C,H,W]
        return torch.sum(attn.unsqueeze(2) * outputs, dim=1)


class MixConv2d(nn.Module):
    """混合核卷积，用于深度卷积层"""

    def __init__(self, in_ch, out_ch, kernels=(3, 5), groups=4):
        super().__init__()
        self.groups = groups
        split_in = [in_ch // groups] * groups
        split_out = [out_ch // groups] * groups

        self.convs = nn.ModuleList()
        for k in kernels:
            self.convs.append(
                nn.Conv2d(sum(split_in[:len(kernels)]),
                          sum(split_out[:len(kernels)]),
                          kernel_size=k, padding=k // 2,
                          groups=groups // len(kernels))
            )

    def forward(self, x):
        features = torch.split(x, x.size(1) // self.groups, dim=1)
        outputs = []
        for i, conv in enumerate(self.convs):
            start = i * (self.groups // len(self.convs))
            end = (i + 1) * (self.groups // len(self.convs))
            inputs = torch.cat(features[start:end], dim=1)
            outputs.append(conv(inputs))
        return torch.cat(outputs, dim=1)


class RepGhostBottleneck(nn.Module):
    """重参数化Ghost模块，用于瓶颈层"""

    def __init__(self, inp, hidden, oup, kernel_size=3):
        super().__init__()
        self.cheap = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size,
                      padding=kernel_size // 2, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True)
        )
        self.primary = nn.Conv2d(inp, hidden, 1, bias=False)

    def forward(self, x):
        x = self.primary(x)
        return x + self.cheap(x)


class PosePostProcessor(nn.Module):
    """关键点后处理模块"""

    def __init__(self, num_keypoints=17, max_people=30, heat_thresh=0.1):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.max_people = max_people
        self.heat_thresh = heat_thresh

    @staticmethod
    def _nms(heatmap, kernel=3):
        pad = (kernel - 1) // 2
        hmax = F.max_pool2d(heatmap, kernel, stride=1, padding=pad)
        keep = (hmax == heatmap).float()
        return heatmap * keep

    def forward(self, heatmaps):
        B, C, H, W = heatmaps.shape
        output = []

        for b in range(B):
            people = []
            for c in range(C):
                heatmap = heatmaps[b, c]
                nms_heat = self._nms(heatmap.unsqueeze(0))

                # 寻找峰值
                y, x = torch.where(nms_heat[0] > self.heat_thresh)
                scores = nms_heat[0, y, x]

                # 按分数排序
                indices = torch.argsort(scores, descending=True)[:self.max_people]
                kps = torch.stack([x[indices], y[indices]], dim=1) * 4  # 缩放坐标到原图
                people.append(kps)

            # 简单分组逻辑
            grouped = [torch.stack([people[k][i] for k in range(C)], dim=0)
                       for i in range(min(len(p) for p in people))]
            output.append([kps.cpu().numpy() for kps in grouped])

        return output


class SegmentKeypointModel(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        self.features = backbone.features

        # 选取合适的层索引
        self.out_indices = [3, 6, 13]
        self.out_channels = [self.features[i].out_channels for i in self.out_indices]
        in_channels = sum(self.out_channels)

        # 通道融合
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # 分割头
        self.seg_head = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )

        # 姿态头
        self.pose_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 55, 1)
        )

        self.postprocessor = PosePostProcessor()

    def forward(self, x):
        # 1) Backbone 特征提取
        feats = []
        h = x
        for idx, layer in enumerate(self.features):
            h = layer(h)
            if idx in self.out_indices:
                feats.append(h)

        # 2) 上采样到同一空间尺寸并拼接
        target_size = feats[0].shape[2:]
        feats = [
            F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
            for f in feats
        ]
        fused = torch.cat(feats, dim=1)
        fused = self.fuse_conv(fused)

        # 3) 分割头：先预测低分辨率 logits，再上采样到输入图尺寸
        seg_logits = self.seg_head(fused)
        seg_logits = F.interpolate(
            seg_logits,
            size=(x.shape[2], x.shape[3]),
            mode='bilinear',
            align_corners=False
        )

        # 4) 姿态头：始终返回 heatmap 张量（没有在 forward 内做 postprocess）
        pose_logits = self.pose_head(fused)
        # 将 heatmap 上采样到 GT heatmap 所需分辨率
        # 假设 postprocessor.stride 表示特征图到原图的步幅
        out_h = x.shape[2] // self.postprocessor.stride
        out_w = x.shape[3] // self.postprocessor.stride
        pose_pred = F.interpolate(
            pose_logits,
            size=(out_h, out_w),
            mode='bilinear',
            align_corners=False
        )

        return seg_logits, pose_pred



if __name__ == "__main__":
    model = SegmentKeypointModel()
    dummy_input = torch.randn(2, 3, 256, 256)
    seg_logits, multi_kps = model(dummy_input)

    print(f"Segmentation logits shape: {seg_logits.shape}")
    print(f"Keypoints structure: {len(multi_kps)} batches")
    print(f"First batch detected people: {len(multi_kps[0])}")
    print(f"Single person keypoints shape: {multi_kps[0][0].shape if len(multi_kps[0]) > 0 else None}")
