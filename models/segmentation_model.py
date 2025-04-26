import torch
import torch.nn as nn
import torchvision.models as models

class UNetSegmentationModel(nn.Module):
    def __init__(self, num_classes=1, encoder_name="resnet34"):
        """
        U-Net model for segmentation.
        num_classes: output channels (for binary human segmentation, 1 or 2 classes)
        encoder_name: choose encoder backbone ('resnet34' or 'resnet50' or None for plain conv encoder)
        """
        super().__init__()
        self.num_classes = num_classes

        # Encoder (backbone)
        if encoder_name.startswith("resnet"):
            # Use a ResNet backbone from torchvision
            resnet = getattr(models, encoder_name)(pretrained=True)
            # Extract layers for feature maps at multiple scales
            # For U-Net, we'll use outputs of conv1, layer1, layer2, layer3, layer4
            self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)          # out: 64 x /2
            self.enc1 = nn.Sequential(resnet.maxpool, resnet.layer1)                 # out: 64 x /4
            self.enc2 = resnet.layer2                                               # out: 128 x /8
            self.enc3 = resnet.layer3                                               # out: 256 x /16
            self.enc4 = resnet.layer4                                               # out: 512 x /32 (for resnet34; 2048 for resnet50)
            encoder_channels = [64, 64, 128, 256, 512] if encoder_name == "resnet34" else [64, 256, 512, 1024, 2048]
        else:
            # If no pretrained backbone, define a simple conv encoder (not recommended for accuracy)
            encoder_channels = [64, 128, 256, 512]
            self.enc1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
            )  # /2 down
            self.pool1 = nn.MaxPool2d(2)
            self.enc2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
            )  # /4 down
            self.pool2 = nn.MaxPool2d(2)
            self.enc3 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)
            )  # /8 down
            self.pool3 = nn.MaxPool2d(2)
            self.enc4 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True)
            )  # /16 down
            self.pool4 = nn.MaxPool2d(2)
            # Note: This plain encoder doesn't go as deep (output /16). Could add another layer for /32 if needed.

        # Decoder (upsampling path)
        # If using ResNet encoder, use conv transpose to upsample from deepest layer and merge with skip connections
        if encoder_name.startswith("resnet"):
            ch4 = encoder_channels[-1]   # e.g., 512 or 2048
            ch3 = encoder_channels[-2]   # e.g., 256 or 1024
            ch2 = encoder_channels[-3]   # e.g., 128 or 512
            ch1 = encoder_channels[-4]   # e.g., 64 or 256
            ch0 = encoder_channels[-5]   # e.g., 64 or 64
            # Upsample 32->16
            self.up4 = nn.ConvTranspose2d(ch4, ch3, kernel_size=2, stride=2)
            self.dec4 = nn.Sequential(
                nn.Conv2d(ch3 + ch3, ch3, kernel_size=3, padding=1), nn.BatchNorm2d(ch3), nn.ReLU(inplace=True),
                nn.Conv2d(ch3, ch3, kernel_size=3, padding=1), nn.BatchNorm2d(ch3), nn.ReLU(inplace=True)
            )
            # Upsample 16->8
            self.up3 = nn.ConvTranspose2d(ch3, ch2, kernel_size=2, stride=2)
            self.dec3 = nn.Sequential(
                nn.Conv2d(ch2 + ch2, ch2, kernel_size=3, padding=1), nn.BatchNorm2d(ch2), nn.ReLU(inplace=True),
                nn.Conv2d(ch2, ch2, kernel_size=3, padding=1), nn.BatchNorm2d(ch2), nn.ReLU(inplace=True)
            )
            # Upsample 8->4
            self.up2 = nn.ConvTranspose2d(ch2, ch1, kernel_size=2, stride=2)
            self.dec2 = nn.Sequential(
                nn.Conv2d(ch1 + ch1, ch1, kernel_size=3, padding=1), nn.BatchNorm2d(ch1), nn.ReLU(inplace=True),
                nn.Conv2d(ch1, ch1, kernel_size=3, padding=1), nn.BatchNorm2d(ch1), nn.ReLU(inplace=True)
            )
            # Upsample 4->2
            self.up1 = nn.ConvTranspose2d(ch1, ch0, kernel_size=2, stride=2)
            self.dec1 = nn.Sequential(
                nn.Conv2d(ch0 + ch0, ch0, kernel_size=3, padding=1), nn.BatchNorm2d(ch0), nn.ReLU(inplace=True),
                nn.Conv2d(ch0, ch0, kernel_size=3, padding=1), nn.BatchNorm2d(ch0), nn.ReLU(inplace=True)
            )
        else:
            # Plain conv encoder case (with 4 pools), we have enc1..enc4 and pools
            self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
            self.dec4 = nn.Sequential(
                nn.Conv2d(256+256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)
            )
            self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            self.dec3 = nn.Sequential(
                nn.Conv2d(128+128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
            )
            self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.dec2 = nn.Sequential(
                nn.Conv2d(64+64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
            )
            self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
            # Note: for plain encoder, enc0 isn't defined (no initial conv separate), we started at enc1.
            self.dec1 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
            )

        # Final segmentation head
        self.final_conv = nn.Conv2d(ch0 if encoder_name.startswith("resnet") else 64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder forward
        if hasattr(self, 'pool1'):  # plain conv encoder path
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool1(e1))
            e3 = self.enc3(self.pool2(e2))
            e4 = self.enc4(self.pool3(e3))
        else:  # ResNet encoder path
            e0 = self.enc0(x)
            e1 = self.enc1(e0)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            e4 = self.enc4(e3)
        # Decoder forward with skip connections
        if hasattr(self, 'pool1'):
            d4 = self.dec4(torch.cat([self.up4(e4), e3], dim=1))
            d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))
            d1 = self.dec1(self.up1(d2))
        else:
            d4 = self.dec4(torch.cat([self.up4(e4), e3], dim=1))
            d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e0], dim=1))
        out = self.final_conv(d1)
        return out
