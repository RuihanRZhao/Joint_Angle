import torch
import torch.nn as nn
from timm import create_model

class ViTPoseModel(nn.Module):
    def __init__(self,
                 backbone_name: str = "vit_base_patch16_224",
                 pretrained: bool = True,
                 img_size: int = 480,
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
        Args
        ----
        img4 : Tensor  # 4-通道 (RGB+mask), [B,4,H,W]  H=W=480
        Returns
        -------
        heatmaps : Tensor [B, K,   H/4, W/4]
        pafs     : Tensor [B, 2*L, H/4, W/4]
        """
        B, _, H, W = img4.shape  # H=W=480

        # 1. Patch embedding  → [B, C, H/16, W/16]
        x = self.backbone.patch_embed(img4)

        # 2. Flatten → [B, N, C]
        x = x.flatten(2).transpose(1, 2).contiguous()

        # 3. prepend cls token
        cls_tok = self.backbone.cls_token.expand(B, -1, -1)      # [B,1,C]
        x = torch.cat((cls_tok, x), dim=1)                       # [B,1+N,C]

        # 4. interpolate pos-embed to 480×480 (30×30 patches)
        pos_embed = self.backbone.pos_embed
        if pos_embed.shape[1] != x.shape[1]:
            cls_pe, patch_pe = pos_embed[:, :1], pos_embed[:, 1:]           # [1,1,C], [1,N0,C]
            Hp = Wp = int(patch_pe.shape[1] ** 0.5)                         # 14
            patch_pe = patch_pe.reshape(1, Hp, Wp, -1).permute(0, 3, 1, 2)  # [1,C,14,14]
            new_Hp, new_Wp = H // 16, W // 16                               # 30,30
            patch_pe = F.interpolate(patch_pe, size=(new_Hp, new_Wp),
                                     mode='bicubic', align_corners=False)
            patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, -1, cls_pe.shape[-1])
            pos_embed = torch.cat([cls_pe, patch_pe], dim=1)                # [1,1+N,C]

        x = x + pos_embed
        x = self.backbone.pos_drop(x)

        # 5. Transformer encoder
        for blk in self.backbone.blocks:
            x = blk(x)
        x = self.backbone.norm(x)

        # 6. Remove cls, reshape back to feature map
        x = x[:, 1:, :]
        h16, w16 = H // 16, W // 16
        feat = x.transpose(1, 2).contiguous().view(B, -1, h16, w16)  # [B,C,H/16,W/16]

        # 7. Decoder → H/4
        up_feats = self.decoder(feat)                  # [B,C',H/4,W/4]

        heatmaps = self.heatmap_head(up_feats)         # [B,K,H/4,W/4]
        pafs     = self.paf_head(up_feats)             # [B,2L,H/4,W/4]

        return heatmaps, pafs