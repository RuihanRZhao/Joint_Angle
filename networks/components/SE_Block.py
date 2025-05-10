import torch.nn as nn

# Squeeze-and-Excitation (SE) block for channel attention
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        # Hidden dimension for squeeze
        hidden_dim = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden_dim, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden_dim, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # Squeeze
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y