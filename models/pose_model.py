import torch
import torch.nn as nn
from timm import create_model

class ViTPoseModel(nn.Module):
    def __init__(self,
                 backbone_name: str = "vit_base_patch16_256",
                 pretrained: bool = True,
                 img_size: int = 256,
                 num_keypoints: int = 17,
                 num_pafs: int = 19):
        super().__init__()
        # timm.create_model(pretrained=True) 会自动下载 ImageNet 权重
        self.backbone = create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=4,
            img_size=img_size,
            num_classes=0
        )
        embed_dim = self.backbone.embed_dim  # e.g. 768 for base

        # 2) 一个简单的 upsampling decoder，将 transformer 特征还原到 H/4
        #    先把序列特征重塑成 [B, C, H/16, W/16]，再多级上采样
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # H/16 -> H/8
            nn.Conv2d(embed_dim, embed_dim//2, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # H/8 -> H/4
            nn.Conv2d(embed_dim//2, embed_dim//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim//4),
            nn.ReLU(inplace=True),
        )

        # 3) 分别为热图和 PAF 定义头
        self.heatmap_head = nn.Conv2d(embed_dim//4, num_keypoints, kernel_size=1)
        self.paf_head     = nn.Conv2d(embed_dim//4, 2 * num_pafs,  kernel_size=1)

    def forward(self, img4: torch.Tensor):
        """
        img4: [B,4,H,W]，4 通道输入（RGB+seg_mask）
        返回：
            heatmaps: [B,K,H/4,W/4]
            pafs:     [B,2*L,H/4,W/4]
        """
        B, C, H, W = img4.shape
        # 1) ViT 前向：得到 [B, N, embed_dim]
        x = self.backbone.patch_embed(img4)              # [B, embed_dim, H/16, W/16]
        x = x.flatten(2).transpose(1,2)                  # [B, N, embed_dim]
        cls_and_patch = torch.cat((self.backbone.cls_token.expand(B, -1, -1), x), dim=1)
        for blk in self.backbone.blocks:
            cls_and_patch = blk(cls_and_patch)
        x = self.backbone.norm(cls_and_patch)[:, 1:, :]  # [B, N, embed_dim]
        # 2) 重塑为特征图
        h16 = int(H // 16)
        w16 = int(W // 16)
        feat = x.transpose(1,2).view(B, -1, h16, w16)    # [B,embed_dim,H/16,W/16]

        # 3) Decoder 上采样到 H/4
        up_feats = self.decoder(feat)                    # [B, embed_dim//4, H/4, W/4]

        # 4) 生成 heatmaps 和 pafs
        heatmaps = self.heatmap_head(up_feats)           # [B,K,H/4,W/4]
        pafs     = self.paf_head(up_feats)               # [B,2*L,H/4,W/4]

        return heatmaps, pafs