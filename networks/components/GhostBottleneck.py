import torch.nn as nn

from .Ghost import GhostModule
from .SE_Block import SEBlock

# Ghost bottleneck block with optional SE (used in high-res branch)
class GhostBottleneck(nn.Module):
    def __init__(self, inp, mid_channels, oup, stride=1, use_se=False):
        super(GhostBottleneck, self).__init__()
        self.stride = stride
        # Pointwise ghost module (expansion)
        self.ghost1 = GhostModule(inp, mid_channels, relu=True)
        # Depthwise convolution for stride (if downsampling)
        self.conv_dw = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, groups=mid_channels, bias=False) if stride > 1 else None
        self.bn_dw = nn.BatchNorm2d(mid_channels) if stride > 1 else None
        # Squeeze-and-Excitation
        self.se = SEBlock(mid_channels) if use_se else nn.Identity()
        # Pointwise linear ghost module (projection)
        self.ghost2 = GhostModule(mid_channels, oup, relu=False)
        # Shortcut connection
        if stride == 1 and inp == oup:
            self.shortcut = nn.Identity()
        else:
            # If stride or channel differs, use depthwise + pointwise conv for skip path
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size=3, stride=stride, padding=1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, oup, kernel_size=1, bias=False),
                nn.BatchNorm2d(oup),
            )
    def forward(self, x):
        y = self.ghost1(x)
        if self.conv_dw is not None:
            y = self.conv_dw(y)
            y = self.bn_dw(y)
        y = self.se(y)
        y = self.ghost2(y)
        # Residual connection
        return y + self.shortcut(x)