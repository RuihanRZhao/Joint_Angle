"""
segmentation.py: Defines the segmentation network.
This network architecture is kept unchanged from the original design (e.g., U-Net).
"""

import torch
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor

class SegFormerSegmentation(torch.nn.Module):
    def __init__(self,
                 pretrained_model_name: str = "nvidia/segformer-b3-finetuned-ade-512-512",
                 num_classes: int = 1):
        super().__init__()
        # 自动处理 resize/normalize/pad
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained(pretrained_model_name)
        self.backbone = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

    def forward(self, x: torch.Tensor):
        # x: [B,3,H,W], float tensor in [0,1]
        imgs = (x * 255).permute(0,2,3,1).cpu().numpy().astype('uint8')
        inputs = self.feature_extractor(images=imgs, return_tensors="pt").to(x.device)
        outputs = self.backbone(**inputs).logits  # [B, num_classes, H, W]
        return outputs