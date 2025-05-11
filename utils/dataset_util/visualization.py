import torch
import wandb
import torchvision
import numpy as np

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
    # ---- 1) 处理 image ----
    # 如果是 numpy，先从 HWC 转为 CHW 的 torch.Tensor
    if isinstance(image_tensor, np.ndarray):
        img_tensor = torch.from_numpy(image_tensor).permute(2, 0, 1)  # HWC -> CHW
    else:
        img_tensor = image_tensor

    # 如果多了 batch 维度（1,3,H,W），去掉它
    if isinstance(img_tensor, torch.Tensor) and img_tensor.ndim == 4:
        img_tensor = img_tensor.squeeze(0)

    # 最终应为 3 维 (C, H, W)
    if not (isinstance(img_tensor, torch.Tensor) and img_tensor.ndim == 3):
        raise ValueError(f"draw_pose_on_image: 期望 image 为 3 维张量 (C,H,W)，但得到 {img_tensor.ndim} 维。")

    # ---- 2) 处理 keypoints ----
    if isinstance(keypoints, np.ndarray):
        kps_tensor = torch.from_numpy(keypoints)
    else:
        kps_tensor = keypoints

    # 如果多了 batch 维度 (1, num_joints, 3)，去掉它
    if isinstance(kps_tensor, torch.Tensor) and kps_tensor.ndim == 3 and kps_tensor.shape[0] == 1:
        kps_tensor = kps_tensor.squeeze(0)

    # 最终应为 [num_joints, 3]
    if not (isinstance(kps_tensor, torch.Tensor) and kps_tensor.ndim == 2 and kps_tensor.shape[1] == 3):
        raise ValueError(f"draw_pose_on_image: 期望 keypoints 为 [num_joints,3]，但得到形状 {tuple(kps_tensor.shape)}。")

    drawn = torchvision.utils.draw_keypoints(
        img_tensor,
        kps_tensor,
        connectivity=COCO_SKELETON,
        colors=color,
        radius=radius,
        width=width
    )
    img_np = drawn[0].permute(1, 2, 0).cpu().numpy()
    return wandb.Image(img_np)