import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import wandb
from typing import Tuple, List


COCO_PERSON_SKELETON: List[Tuple[int, int]] = [
    (15, 13), (13, 11), (16, 14), (14, 12),
    (5, 11), (6, 12), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1),
    (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)
]


NUM_KP = 17
NUM_LIMBS = len(COCO_PERSON_SKELETON)

class InvertedResidual(nn.Module):
    """MobileNetV2的倒残差模块 (薄->厚->薄, 深度可分离卷积)"""
    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        self.use_res_connect = (stride == 1 and in_ch == out_ch)
        hidden_dim = int(round(in_ch * expand_ratio))
        layers = []
        # 扩张卷积 (pointwise 1x1) 提升通道数
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_ch, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        # 深度卷积 (depthwise 3x3)
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))
        # 压缩卷积 (pointwise 1x1 linear) 将通道降至out_ch
        layers.append(nn.Conv2d(hidden_dim, out_ch, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_ch))
        self.conv = nn.Sequential(*layers)
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2Backbone(nn.Module):
    """MobileNetV2骨干网络，用于特征提取"""
    def __init__(self, width_mult=1.0):
        super().__init__()
        # 根据宽度倍率计算各层通道数，保证可被8整除
        def _make_divisible(v, divisor=8):
            return int(math.ceil(v / divisor) * divisor)
        # 定义各阶段通道
        ch1 = _make_divisible(32 * width_mult, 8)
        ch2 = _make_divisible(16 * width_mult, 8)
        ch3 = _make_divisible(24 * width_mult, 8)
        ch4 = _make_divisible(32 * width_mult, 8)
        ch5 = _make_divisible(64 * width_mult, 8)
        ch6 = _make_divisible(96 * width_mult, 8)
        ch7 = _make_divisible(160 * width_mult, 8)
        ch8 = _make_divisible(320 * width_mult, 8)
        # 构建MobileNetV2各层
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, ch1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch1),
            nn.ReLU6(inplace=True)
        )
        # 第一阶段
        self.block1 = InvertedResidual(ch1, ch2, stride=1, expand_ratio=1)
        # 第二阶段 (降采样至1/4)
        self.block2 = InvertedResidual(ch2, ch3, stride=2, expand_ratio=6)
        self.block3 = InvertedResidual(ch3, ch3, stride=1, expand_ratio=6)
        # 第三阶段 (降采样至1/8)
        self.block4 = InvertedResidual(ch3, ch4, stride=2, expand_ratio=6)
        self.block5 = InvertedResidual(ch4, ch4, stride=1, expand_ratio=6)
        self.block6 = InvertedResidual(ch4, ch4, stride=1, expand_ratio=6)
        # 第四阶段 (降采样至1/16)
        self.block7 = InvertedResidual(ch4, ch5, stride=2, expand_ratio=6)
        self.block8 = InvertedResidual(ch5, ch5, stride=1, expand_ratio=6)
        self.block9 = InvertedResidual(ch5, ch5, stride=1, expand_ratio=6)
        self.block10 = InvertedResidual(ch5, ch5, stride=1, expand_ratio=6)
        # 第五阶段 (保持1/16, 提升通道)
        self.block11 = InvertedResidual(ch5, ch6, stride=1, expand_ratio=6)
        self.block12 = InvertedResidual(ch6, ch6, stride=1, expand_ratio=6)
        self.block13 = InvertedResidual(ch6, ch6, stride=1, expand_ratio=6)
        # 第六阶段 (降采样至1/32)
        self.block14 = InvertedResidual(ch6, ch7, stride=2, expand_ratio=6)
        self.block15 = InvertedResidual(ch7, ch7, stride=1, expand_ratio=6)
        self.block16 = InvertedResidual(ch7, ch7, stride=1, expand_ratio=6)
        # 第七阶段 (保持1/32, 扩展通道)
        self.block17 = InvertedResidual(ch7, ch8, stride=1, expand_ratio=6)
    def forward(self, x):
        # 前向传播提取多尺度特征
        x = self.conv1(x)          # 下采样1/2
        x = self.block1(x)         # 1/2 -> 通道ch2
        x = self.block2(x)         # 下采样至1/4
        x = self.block3(x)         # 1/4 -> 通道ch3
        feat2 = x                  # 保存1/4特征
        x = self.block4(x)         # 下采样至1/8
        x = self.block5(x)
        x = self.block6(x)         # 1/8 -> 通道ch4
        feat3 = x                  # 保存1/8特征
        x = self.block7(x)         # 下采样至1/16
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)        # 1/16 -> 通道ch5
        feat5 = x                  # 保存1/16特征
        x = self.block11(x)        # 保持1/16
        x = self.block12(x)
        x = self.block13(x)        # 1/16 -> 通道ch6
        x = self.block14(x)        # 下采样至1/32
        x = self.block15(x)
        x = self.block16(x)        # 1/32 -> 通道ch7
        x = self.block17(x)        # 1/32 -> 通道ch8
        feat7 = x                  # 保存1/32特征
        return feat2, feat3, feat5, feat7

class MultiPoseNet(nn.Module):
    def __init__(self, num_keypoints=17, width_mult=1.0, refine=True):
        super().__init__()
        self.refine = refine
        # 骨干网络
        self.backbone = MobileNetV2Backbone(width_mult=width_mult)

        # 融合通道数
        unify_dim = 128 if width_mult <= 1.0 else int(math.ceil(128 * width_mult / 8) * 8)

        # 1/32 尺度特征 → unify_dim
        self.conv_f7 = nn.Conv2d(
            int(math.ceil(320 * width_mult / 8) * 8),
            unify_dim,
            kernel_size=1,
            bias=False
        )
        self.bn_f7 = nn.BatchNorm2d(unify_dim)

        # 1/16 尺度特征 → unify_dim
        in_ch_f5 = int(math.ceil(64 * width_mult / 8) * 8)
        self.conv_f5 = nn.Conv2d(in_ch_f5, unify_dim, kernel_size=1, bias=False)
        self.bn_f5   = nn.BatchNorm2d(unify_dim)

        # 1/8 尺度特征 → unify_dim
        in_ch_f3 = int(math.ceil(32 * width_mult / 8) * 8)
        self.conv_f3 = nn.Conv2d(in_ch_f3, unify_dim, kernel_size=1, bias=False)
        self.bn_f3   = nn.BatchNorm2d(unify_dim)

        # 1/4 尺度特征 → unify_dim
        in_ch_f2 = int(math.ceil(24 * width_mult / 8) * 8)
        self.conv_f2 = nn.Conv2d(in_ch_f2, unify_dim, kernel_size=1, bias=False)
        self.bn_f2   = nn.BatchNorm2d(unify_dim)

        # 融合后平滑卷积
        self.smooth1 = nn.Conv2d(unify_dim, unify_dim, kernel_size=3, padding=1, bias=False)
        self.bn_s1   = nn.BatchNorm2d(unify_dim)
        self.smooth2 = nn.Conv2d(unify_dim, unify_dim, kernel_size=3, padding=1, bias=False)
        self.bn_s2   = nn.BatchNorm2d(unify_dim)
        self.smooth3 = nn.Conv2d(unify_dim, unify_dim, kernel_size=3, padding=1, bias=False)
        self.bn_s3   = nn.BatchNorm2d(unify_dim)

        # Heatmap & PAF 输出
        self.heatmap_head = nn.Conv2d(unify_dim, num_keypoints, 1)
        self.paf_head     = nn.Conv2d(unify_dim, 2 * NUM_LIMBS, 1)

        # Refinement 模块（如启用）
        if refine:
            refine_in = unify_dim + num_keypoints + 2 * NUM_LIMBS
            self.refine_conv1 = nn.Conv2d(refine_in, unify_dim, 3, padding=1, bias=False)
            self.bn_ref1      = nn.BatchNorm2d(unify_dim)
            self.refine_conv2 = nn.Conv2d(unify_dim, unify_dim, 3, padding=1, bias=False)
            self.bn_ref2      = nn.BatchNorm2d(unify_dim)
            self.refine_heatmap = nn.Conv2d(unify_dim, num_keypoints, 1)
            self.refine_paf     = nn.Conv2d(unify_dim, 2 * NUM_LIMBS, 1)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 1. 骨干特征提取 (多尺度)
        feat2, feat3, feat5, feat7 = self.backbone(x)
        # 2. 特征融合 (自顶向下, 类似FPN)
        f7 = F.relu6(self.bn_f7(self.conv_f7(feat7)))
        f5 = F.relu6(self.bn_f5(self.conv_f5(feat5)))
        f7_up = F.interpolate(f7, size=f5.shape[2:], mode='bilinear', align_corners=False)
        merge_5 = f7_up + f5  # 融合1/16尺度
        merge_5 = F.relu6(self.bn_s1(self.smooth1(merge_5)))
        f3 = F.relu6(self.bn_f3(self.conv_f3(feat3)))
        merge_5_up = F.interpolate(merge_5, size=f3.shape[2:], mode='bilinear', align_corners=False)
        merge_3 = merge_5_up + f3  # 融合1/8尺度
        merge_3 = F.relu6(self.bn_s2(self.smooth2(merge_3)))
        f2 = F.relu6(self.bn_f2(self.conv_f2(feat2)))
        merge_3_up = F.interpolate(merge_3, size=f2.shape[2:], mode='bilinear', align_corners=False)
        merge_2 = merge_3_up + f2  # 融合1/4尺度
        merge_2 = F.relu6(self.bn_s3(self.smooth3(merge_2)))
        # 3. 初始Heatmap和PAF输出
        init_heatmap = self.heatmap_head(merge_2)
        init_paf = self.paf_head(merge_2)
        if not self.refine:
            return init_heatmap, init_paf
        # 4. 精细化阶段: 融合初始输出进行二次预测
        refine_input = torch.cat([merge_2, init_heatmap, init_paf], dim=1)
        r = self.relu(self.bn_ref1(self.refine_conv1(refine_input)))
        r = self.relu(self.bn_ref2(self.refine_conv2(r)))
        refined_heatmap = self.refine_heatmap(r)
        refined_paf = self.refine_paf(r)
        # 返回 (精细化热图, 精细化PAF)


        return refined_heatmap, refined_paf, init_heatmap, init_paf
