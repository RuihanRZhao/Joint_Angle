import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large


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
        backbone = mobilenet_v3_large(pretrained=True)
        self.features = backbone.features

        # 结构优化
        self._modify_bottlenecks()
        self._replace_attention()

        # 分割头
        self.seg_head = nn.Sequential(
            nn.Conv2d(160, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )

        # 姿态头
        self.pose_head = nn.Sequential(
            nn.Conv2d(160, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 17, 1)
        )

        self.postprocessor = PosePostProcessor()

    def _modify_bottlenecks(self):
        # 替换前3个瓶颈块为RepGhost
        for i in [3, 6, 9]:
            block = self.features[i].block
            # block[0] 是 Conv2dNormActivation，它的第一个子模块才是 Conv2d
            conv0 = block[0][0]
            inp = conv0.in_channels
            # block[0].out_channels 对应扩张通道数（hidden dim）
            hidden = block[0].out_channels
            # block[-1] 通常是输出那条 1×1 卷积
            oup = block[-1].out_channels

            new_bottleneck = RepGhostBottleneck(inp, hidden, oup)
            # 用我们的重参数化瓶颈替换掉原来的第一个 Conv
            self.features[i].block[0] = new_bottleneck


    def _replace_attention(self):
        # 替换SE模块为CoordinateAttention
        for i in [1, 4, 7, 10, 13, 16]:
            if hasattr(self.features[i], 'attention'):
                in_ch = self.features[i].attention.se[1].in_channels
                self.features[i].attention = CoordinateAttention(in_ch)

    def forward(self, x):
        # 特征提取
        features = self.features(x)

        # 分割输出
        seg_logits = F.interpolate(
            self.seg_head(features),
            scale_factor=32,
            mode='bilinear',
            align_corners=True
        )

        # 姿态热图
        heatmaps = F.interpolate(
            self.pose_head(features),
            scale_factor=32,
            mode='bilinear',
            align_corners=True
        )

        # 关键点解码
        multi_kps = self.postprocessor(heatmaps)

        return seg_logits, multi_kps


if __name__ == "__main__":
    model = SegmentKeypointModel()
    dummy_input = torch.randn(2, 3, 256, 256)
    seg_logits, multi_kps = model(dummy_input)

    print(f"Segmentation logits shape: {seg_logits.shape}")
    print(f"Keypoints structure: {len(multi_kps)} batches")
    print(f"First batch detected people: {len(multi_kps[0])}")
    print(f"Single person keypoints shape: {multi_kps[0][0].shape if len(multi_kps[0]) > 0 else None}")
