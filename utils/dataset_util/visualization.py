import numpy as np
import cv2
import torch

# COCO 17 keypoints skeleton connections
COCO_PERSON_SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (0,5),(0,6),(5,7),(7,9),
    (6,8),(8,10),(5,6),(11,12),
    (5,11),(6,12),(11,13),(13,15),
    (12,14),(14,16)
]


def draw_pose_on_image(image, keypoints_list, color=None, threshold=0.05):
    """
    在图像上绘制人体关键点和骨骼连线，兼容 COCO gt/dt 格式。

    Args:
        image (np.ndarray or torch.Tensor): 原始图像，RGB 或 BGR 格式。
        keypoints_list (list or np.ndarray): 长度为 3*J 的浮点列表或数组，格式 [x1,y1,v1,...,xJ,yJ,vJ]。
        color (tuple): 绘制颜色，BGR 格式，默认为绿色 (0,255,0)。
        threshold (float): 置信度阈值，低于该值的关键点/连线不绘制。

    Returns:
        np.ndarray: 绘制后的图像，BGR 格式，dtype=uint8。
    """
    # 将 torch.Tensor 转为 numpy
    if isinstance(image, torch.Tensor):
        img_np = image.cpu().numpy()
        img_np = (img_np.transpose(1, 2, 0) * 255).astype(np.uint8)
    else:
        img_np = image.copy()
        if img_np.dtype != np.uint8:
            img_np = img_np.astype(np.uint8)
    # 确保是 BGR，如果是 RGB，则转换
    if img_np.shape[2] == 3 and img_np.dtype == np.uint8:
        # 判断是否为 RGB (可以通过色彩分量判断或用户保证)
        # 我们假设传入的是 BGR 或 RGB，用户自行处理
        pass

    # 处理 keypoints_list
    kps = np.array(keypoints_list, dtype=np.float32)
    if kps.ndim == 1:
        kps = kps.reshape(-1, 3)
    # 默认颜色
    if color is None:
        color = (0, 255, 0)

    # 绘制骨骼连线
    for i, j in COCO_PERSON_SKELETON:
        if kps[i, 2] > threshold and kps[j, 2] > threshold:
            x1, y1 = int(kps[i, 0]), int(kps[i, 1])
            x2, y2 = int(kps[j, 0]), int(kps[j, 1])
            cv2.line(img_np, (x1, y1), (x2, y2), color, 2)
    # 绘制关键点
    for x, y, v in kps:
        if v > threshold:
            cv2.circle(img_np, (int(x), int(y)), 3, color, -1)

    return img_np