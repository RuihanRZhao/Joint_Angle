import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, act=True):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels, kernel_size=3)
        self.conv2 = ConvBNAct(channels, channels, kernel_size=3, act=False)

    def forward(self, x):
        return F.silu(self.conv2(self.conv1(x)) + x)

class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n=2, expansion=0.5, use_se=True):
        super().__init__()
        hidden = int(out_channels * expansion)
        self.conv1 = ConvBNAct(in_channels, hidden, kernel_size=1)
        self.conv2 = ConvBNAct(in_channels, hidden, kernel_size=1)
        self.blocks = nn.Sequential(*[Residual(hidden) for _ in range(n)])
        self.conv3 = ConvBNAct(hidden * 2, out_channels, kernel_size=1)
        self.attn = SEBlock(out_channels) if use_se else nn.Identity()

    def forward(self, x):
        y1 = self.blocks(self.conv1(x))
        y2 = self.conv2(x)
        out = torch.cat([y1, y2], dim=1)
        out = self.conv3(out)
        return self.attn(out)

class JointPoseNet(nn.Module):
    def __init__(self, num_keypoints=17, bins=4, image_size=384):
        super().__init__()
        feat_size = image_size // 4
        self.bins = bins

        self.stem = ConvBNAct(3, 64, 3, stride=2)

        self.down1 = ConvBNAct(64, 128, 3, stride=2)
        self.stage1 = CSPBlock(128, 128, n=2)

        self.down2 = ConvBNAct(128, 256, 3, stride=2)
        self.stage2 = CSPBlock(256, 256, n=4)

        self.down3 = ConvBNAct(256, 512, 3, stride=2)
        self.stage3 = CSPBlock(512, 512, n=4)

        # Fusion
        self.reduce3 = ConvBNAct(512, 256, 1)
        self.fuse2 = ConvBNAct(256, 256, 3)
        self.reduce2 = ConvBNAct(256, 128, 1)
        self.fuse1 = ConvBNAct(128, 128, 3)

        # SimCC Head
        self.conv_x = nn.Conv2d(128, num_keypoints, (feat_size, 1), stride=(feat_size, 1))
        self.conv_y = nn.Conv2d(128, num_keypoints, (1, feat_size), stride=(1, feat_size))

        self.deconv_x = nn.ConvTranspose1d(num_keypoints, num_keypoints, bins, bins, groups=num_keypoints)
        self.deconv_y = nn.ConvTranspose1d(num_keypoints, num_keypoints, bins, bins, groups=num_keypoints)

        nn.init.constant_(self.deconv_x.weight, 1.0)
        nn.init.constant_(self.deconv_y.weight, 1.0)

    def forward(self, x):
        x = self.stem(x)
        s1 = self.stage1(self.down1(x))
        s2 = self.stage2(self.down2(s1))
        s3 = self.stage3(self.down3(s2))

        s3_up = F.interpolate(self.reduce3(s3), size=s2.shape[-2:], mode='nearest')
        f2 = self.fuse2(s3_up + s2)

        f2_up = F.interpolate(self.reduce2(f2), size=s1.shape[-2:], mode='nearest')
        f1 = self.fuse1(f2_up + s1)

        x_feat = self.conv_x(f1).squeeze(2)
        y_feat = self.conv_y(f1).squeeze(3)
        x_logits = self.deconv_x(x_feat)
        y_logits = self.deconv_y(y_feat)

        return x_logits, y_logits
