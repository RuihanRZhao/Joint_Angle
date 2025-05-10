import math
import torch.nn as nn

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        # number of primary convolution output channels
        init_channels = math.ceil(oup / ratio)
        # number of "ghost" channels to generate
        new_channels = oup - init_channels
        # Primary convolution
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride=stride, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )
        # Cheap operation (depthwise conv to generate ghost features)
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, stride=1, padding=dw_size//2, groups=init_channels, bias=False) if new_channels > 0 else nn.Identity(),
            nn.BatchNorm2d(new_channels) if new_channels > 0 else nn.Identity(),
            nn.ReLU(inplace=True) if relu and new_channels > 0 else nn.Identity(),
        )
    def forward(self, x):
        x1 = self.primary_conv(x)
        if isinstance(self.cheap_operation[0], nn.Identity):
            # no ghost channels to generate
            return x1
        x2 = self.cheap_operation(x1)
        # Concatenate primary and ghost
        out = torch.cat([x1, x2], dim=1)
        # In case rounding caused extra channels, trim to desired oup
        return out[:, :self.oup, :, :]