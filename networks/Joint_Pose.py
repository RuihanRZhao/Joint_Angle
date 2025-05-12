import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNAct(nn.Module):
    """Convolution followed by BatchNorm and activation."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, act=True):
        super(ConvBNAct, self).__init__()
        padding = kernel_size // 2  # assume square kernel, simple padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                               padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act else nn.Identity()  # SiLU activation (Swish)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Residual(nn.Module):
    """Basic residual block: 3x3 conv -> 3x3 conv with skip connection."""
    def __init__(self, channels):
        super(Residual, self).__init__()
        self.conv1 = ConvBNAct(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvBNAct(channels, channels, kernel_size=3, stride=1, act=False)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return F.silu(out + x)  # use SiLU for the fused activation

class CSPBlock(nn.Module):
    """
    Cross Stage Partial (CSP) Block: splits input channels, processes one part, then concatenates.
    Args:
        in_channels (int): input channels.
        out_channels (int): output channels.
        n (int): number of residual blocks in the partial path.
        expansion (float): ratio for hidden channels relative to out_channels.
    """
    def __init__(self, in_channels, out_channels, n=1, expansion=0.5):
        super(CSPBlock, self).__init__()
        hidden_channels = int(out_channels * expansion)
        # Split: two conv layers generate two feature parts
        self.conv1 = ConvBNAct(in_channels, hidden_channels, kernel_size=1)
        self.conv2 = ConvBNAct(in_channels, hidden_channels, kernel_size=1)
        # Residual blocks on the first part
        self.m = nn.Sequential(*[Residual(hidden_channels) for _ in range(n)])
        # Concat and fuse
        self.conv3 = ConvBNAct(hidden_channels * 2, out_channels, kernel_size=1)
    def forward(self, x):
        y1 = self.conv1(x)
        y1 = self.m(y1)
        y2 = self.conv2(x)
        out = torch.cat([y1, y2], dim=1)
        return self.conv3(out)

class JointPoseNet(nn.Module):
    """
    JointPoseNet: Pose estimation network with CSPNeXt-style backbone and SimCC head.
    Args:
        num_keypoints (int): number of keypoints (K).
        bins (int): bins per pixel for SimCC (default 4, common choices 2, 4, 10).
        image_size (int): input image size (assumed square, default 384).
    """
    def __init__(self, num_keypoints, bins=4, image_size=384):
        super(JointPoseNet, self).__init__()
        self.num_keypoints = num_keypoints
        self.bins = bins
        # Calculate final feature map size (after upsample fusion, 1/4 of input)
        feat_map_size = image_size // 4  # 384 -> 96
        # Backbone: Stem + Stages with CSP blocks
        self.stem = ConvBNAct(3, 64, kernel_size=3, stride=2)         # 1/2 size
        # Stage 1: 1/4 size
        self.conv_down1 = ConvBNAct(64, 128, kernel_size=3, stride=2) # 1/4 size
        self.csp1 = CSPBlock(128, 128, n=1)
        # Stage 2: 1/8 size
        self.conv_down2 = ConvBNAct(128, 256, kernel_size=3, stride=2) # 1/8 size
        self.csp2 = CSPBlock(256, 256, n=2)
        # Stage 3: 1/16 size
        self.conv_down3 = ConvBNAct(256, 512, kernel_size=3, stride=2) # 1/16 size
        self.csp3 = CSPBlock(512, 512, n=2)
        # Feature fusion modules (upsample and merge higher-level features with lower-level)
        self.reduce_conv3 = ConvBNAct(512, 256, kernel_size=1)  # reduce stage3 channels to stage2's
        self.fuse_conv2   = ConvBNAct(256, 256, kernel_size=3)  # fuse stage3 into stage2
        self.reduce_conv2 = ConvBNAct(256, 128, kernel_size=1)  # reduce fused stage2 to stage1's channels
        self.fuse_conv1   = ConvBNAct(128, 128, kernel_size=3)  # fuse stage2 into stage1 (final feature map)
        # SimCC head: produce separate x and y distributions
        # Conv layers to collapse one spatial dimension:
        self.conv_x = nn.Conv2d(128, num_keypoints, kernel_size=(feat_map_size, 1),
                                 stride=(feat_map_size, 1), bias=True)
        self.conv_y = nn.Conv2d(128, num_keypoints, kernel_size=(1, feat_map_size),
                                 stride=(1, feat_map_size), bias=True)
        # Transposed conv (1D) to upsample collapsed features by 'bins' times for finer classification
        self.conv_transpose_x = nn.ConvTranspose1d(num_keypoints, num_keypoints,
                                                   kernel_size=bins, stride=bins,
                                                   groups=num_keypoints, bias=False)
        self.conv_transpose_y = nn.ConvTranspose1d(num_keypoints, num_keypoints,
                                                   kernel_size=bins, stride=bins,
                                                   groups=num_keypoints, bias=False)
        # Initialize transposed conv kernels to replicate values (near uniform)
        nn.init.constant_(self.conv_transpose_x.weight, 1.0)
        nn.init.constant_(self.conv_transpose_y.weight, 1.0)
        if self.conv_transpose_x.bias is not None:
            nn.init.zeros_(self.conv_transpose_x.bias)
        if self.conv_transpose_y.bias is not None:
            nn.init.zeros_(self.conv_transpose_y.bias)

    def forward(self, x):
        # Backbone forward
        x = self.stem(x)
        stage1 = self.csp1(self.conv_down1(x))    # 1/4
        stage2 = self.csp2(self.conv_down2(stage1))  # 1/8
        stage3 = self.csp3(self.conv_down3(stage2))  # 1/16
        # Feature fusion
        # Upsample stage3 to stage2 resolution and fuse
        stage3_up = F.interpolate(self.reduce_conv3(stage3), size=stage2.shape[-2:], mode='nearest')
        fuse2 = self.fuse_conv2(stage3_up + stage2)
        # Upsample fused stage2 to stage1 resolution and fuse
        fuse2_up = F.interpolate(self.reduce_conv2(fuse2), size=stage1.shape[-2:], mode='nearest')
        fused_final = self.fuse_conv1(fuse2_up + stage1)  # final feature at 1/4 resolution
        # SimCC Head: produce x-axis and y-axis logits
        # X-axis: collapse height dimension
        x_feat = self.conv_x(fused_final)        # [B, K, 1, W_feat]
        x_feat = x_feat.squeeze(2)               # [B, K, W_feat]
        x_logits = self.conv_transpose_x(x_feat) # [B, K, W_feat * bins]
        # Y-axis: collapse width dimension
        y_feat = self.conv_y(fused_final)        # [B, K, H_feat, 1]
        y_feat = y_feat.squeeze(3)               # [B, K, H_feat]
        y_logits = self.conv_transpose_y(y_feat) # [B, K, H_feat * bins]
        return x_logits, y_logits

    def load_pretrained_backbone(self, state_dict_or_path):
        """
        Load pretrained weights for backbone layers (e.g., ImageNet-1K).
        Skips head and fusion layers.
        Args:
            state_dict_or_path: state dict object or path to .pth file.
        """
        # Load the state dictionary
        if isinstance(state_dict_or_path, str):
            state_dict = torch.load(state_dict_or_path, map_location='cpu')
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
        else:
            state_dict = state_dict_or_path
        model_dict = self.state_dict()
        filtered_dict = {}
        for k, v in state_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                # Only load backbone-related layers
                if not (k.startswith('conv_x') or k.startswith('conv_y') or k.startswith('conv_transpose') or
                        k.startswith('fuse_conv') or k.startswith('reduce_conv')):
                    filtered_dict[k] = v
        model_dict.update(filtered_dict)
        self.load_state_dict(model_dict, strict=False)
