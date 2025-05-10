import os
import random
import torch
import wandb
import torchvision
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# COCO 17 骨架连接关系，索引对应 COCO keypoints 顺序
COCO_SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (1,2),(3,5),(4,6),(5,6),
    (5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),(11,13),
    (13,15),(12,14),(14,16)
]

def draw_pose_on_image(
    image_tensor: torch.Tensor,
    keypoints: torch.Tensor,
    color=(0, 255, 0),
    radius: int = 2,
    width: int = 3
) -> wandb.Image:
    """
    在单张图像上绘制 COCO 17 点关键点骨架，并返回 wandb.Image。

    Args:
        image_tensor: torch.Tensor, shape (3, H, W), dtype=uint8, BGR, device='cuda'.
        keypoints: torch.Tensor, shape (17, 3), (x, y, score), device 同上.
        color: Union[str, Tuple[int,int,int]], 骨架颜色，BGR 或颜色名，默认绿色.
        radius: int, 关键点圆点半径.
        width: int, 骨架连线宽度.

    Returns:
        wandb.Image: 绘制后图像，适合 wandb.log。
    """
    if not (isinstance(image_tensor, torch.Tensor) and isinstance(keypoints, torch.Tensor)):
        raise TypeError("image_tensor 和 keypoints 必须为 torch.Tensor 类型")
    if image_tensor.device.type != 'cuda' or keypoints.device.type != 'cuda':
        raise ValueError("输入张量必须放置在 GPU(cuda) 上")
    if image_tensor.ndim != 3 or image_tensor.shape[0] != 3:
        raise ValueError("image_tensor 必须为 (3, H, W) 维度")
    if keypoints.ndim != 2 or keypoints.shape != (17, 3):
        raise ValueError("keypoints 必须为 (17,3) 维度，格式为 (x, y, score)")
    if image_tensor.dtype != torch.uint8:
        raise ValueError("image_tensor 必须为 uint8 类型 (0-255)")

    img_batch = image_tensor.unsqueeze(0)
    kps = keypoints[:, :2].round().long().unsqueeze(0)
    drawn = torchvision.utils.draw_keypoints(
        img_batch,
        kps,
        connectivity=COCO_SKELETON,
        colors=color,
        radius=radius,
        width=width
    )
    img_np = drawn[0].permute(1, 2, 0).cpu().numpy()
    return wandb.Image(img_np)