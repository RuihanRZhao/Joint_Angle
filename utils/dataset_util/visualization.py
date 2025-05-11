from typing import Tuple

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

def _to_tensor_and_meta(image):
    """
    将输入 image 转为 CHW Tensor，并返回元信息：
      - img_tensor: torch.Tensor, C×H×W
      - input_is_numpy: bool, 原始输入是否是 numpy.ndarray
      - batch_dim: bool, 是否原本带有 batch 维度 [1,C,H,W]
    """
    input_is_numpy = isinstance(image, np.ndarray)
    if input_is_numpy:
        # H×W×C -> C×H×W
        img_tensor = torch.from_numpy(image).permute(2, 0, 1)
        batch_dim = False
    else:
        img_tensor = image
        # 若为 [1,C,H,W]，去掉 batch 维度
        if img_tensor.ndim == 4 and img_tensor.shape[0] == 1:
            batch_dim = True
            img_tensor = img_tensor.squeeze(0)
        else:
            batch_dim = False

    if img_tensor.ndim != 3 or img_tensor.shape[0] not in (1, 3):
        raise ValueError(f"_to_tensor_and_meta: 期望 image 为 C×H×W (C=1或3)，但得到 {tuple(img_tensor.shape)}。")

    return img_tensor, input_is_numpy, batch_dim

def _to_keypoint_coords(keypoints):
    """
    将 keypoints 转为 shape 为 [1, K, 2] 的坐标 tensor（去掉置信度，添加 batch 维）。
    支持输入 [K,3]、[1,K,3]、numpy.ndarray 或 torch.Tensor。
    """
    if isinstance(keypoints, np.ndarray):
        kps = torch.from_numpy(keypoints)
    else:
        kps = keypoints

    # 去掉多余 batch 维
    if kps.ndim == 3 and kps.shape[0] == 1:
        kps = kps.squeeze(0)
    if kps.ndim != 2 or kps.shape[1] != 3:
        raise ValueError(f"_to_keypoint_coords: 期望 keypoints 为 [K,3]，但得到 {tuple(kps.shape)}。")

    # 提取 x,y 并扩 batch 维 -> [1, K, 2]
    coords = kps[:, :2].unsqueeze(0)
    return coords

def _draw(img_tensor, coords, color):
    """
    在单张 C×H×W Tensor 上绘制 keypoints（coords: [1,K,2]），
    返回同样 shape 的 Tensor。
    """
    # torchvision.draw_keypoints 要求 img_tensor 维度为 [C,H,W]，coords 为 [num_instances, K, 2]
    drawn = torchvision.utils.draw_keypoints(img_tensor, coords, COCO_SKELETON, colors=color)
    return drawn

def _to_original_format(drawn_tensor, input_is_numpy, batch_dim):
    """
    将绘制后的张量还原成和原始输入相同的格式：
      - 若是 numpy 输入，返回 H×W×C ndarray
      - 若是 tensor 且带 batch 维，则返回 [1,C,H,W]
      - 否则返回 [C,H,W] Tensor
    """
    # 如果原来有 batch 维，先补回
    if batch_dim:
        drawn_tensor = drawn_tensor.unsqueeze(0)

    if input_is_numpy:
        # CHW -> HWC 或 BCHW -> BHWC，然后去除 batch 维
        if drawn_tensor.ndim == 4:
            arr = drawn_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        else:
            arr = drawn_tensor.permute(1, 2, 0).cpu().numpy()
        return arr
    else:
        return drawn_tensor

def draw_pose_on_image(image, keypoints, color=(255, 0, 0)):
    """
    在单张 RGB 图像上绘制人体关键点，返回与输入 image 相同类型和形状的数据。

    Args:
        image:
          - numpy.ndarray (H, W, 3)
          - torch.Tensor (3, H, W) 或 (1, 3, H, W)
        keypoints:
          - numpy.ndarray (K, 3) 或 torch.Tensor (K, 3)
          - 或带 batch 维的 (1, K, 3)
        color: (R, G, B) 整数元组

    Returns:
        与输入 image 格式一致的绘制后图像。
    """
    img_tensor, is_numpy, has_batch = _to_tensor_and_meta(image)
    coords = _to_keypoint_coords(keypoints)
    drawn = _draw(img_tensor, coords, color)
    return _to_original_format(drawn, is_numpy, has_batch)