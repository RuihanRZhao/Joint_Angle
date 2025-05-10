import torch.nn as nn

from SE_Block import SEBlock

# Inverted Residual block (MobileNetV2 bottleneck) with optional SE
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_se=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2], "Stride must be 1 or 2"
        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = (self.stride == 1 and inp == oup)
        layers = []
        # expansion (pointwise conv)
        if expand_ratio != 1:
            layers.append(nn.Conv2d(inp, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        else:
            hidden_dim = inp
        # depthwise convolution
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        if use_se:
            layers.append(SEBlock(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))
        # projection (pointwise linear conv)
        layers.append(nn.Conv2d(hidden_dim, oup, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(oup))
        self.conv = nn.Sequential(*layers)
    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            # Residual connection
            return x + out
        else:
            return out