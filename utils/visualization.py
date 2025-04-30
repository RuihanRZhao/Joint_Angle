"""
visualization_utils.py - Enhanced visualization tools for segmentation and pose estimation
"""

import cv2
import numpy as np
import torch
from typing import Optional, Tuple, List

# COCO关键点连接定义 (17关键点)
COCO_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12),  # 下肢
    (11, 12), (5, 11), (6, 12),              # 躯干
    (5, 7), (7, 9), (6, 8), (8, 10),         # 上肢
    (1, 2), (0, 1), (0, 2),                  # 面部
    (0, 3), (0, 4), (3, 5), (4, 6)           # 肩耳连接
]

def overlay_segmentation(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    colormap: Optional[str] = 'jet',
    custom_color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """
    增强版分割掩码叠加，支持多类别和自定义颜色

    Args:
        image: BGR图像 (H,W,3) uint8
        mask: 分割掩码 (H,W) int32，0表示背景
        alpha: 叠加透明度 [0,1]
        colormap: OpenCV色图名称或None
        custom_color: 自定义颜色 (BGR)，当colormap为None时使用

    Returns:
        叠加后的BGR图像
    """
    # 参数验证
    assert 0 <= alpha <= 1, "Alpha值必须在0到1之间"
    assert image.shape[:2] == mask.shape, "图像和掩码尺寸必须一致"

    # 生成颜色映射
    if colormap:
        if mask.dtype != np.uint8:
            mask_norm = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            mask_norm = mask
        color_mask = cv2.applyColorMap(mask_norm, getattr(cv2, f'COLORMAP_{colormap.upper()}'))
    else:
        color_mask = np.zeros_like(image)
        color_mask[mask > 0] = custom_color

    # 混合图像
    blended = cv2.addWeighted(image, 1-alpha, color_mask, alpha, 0)
    return blended

def draw_pose(
    image: np.ndarray,
    keypoints: np.ndarray,
    confidences: Optional[np.ndarray] = None,
    skeleton: List[Tuple[int, int]] = COCO_SKELETON,
    point_radius: int = 5,
    skeleton_thickness: int = 2,
    color_scheme: str = 'hsv'
) -> np.ndarray:
    """
    绘制带骨架连接的人体姿态

    Args:
        image: BGR图像 (H,W,3) uint8
        keypoints: 关键点坐标 (N,2) float32
        confidences: 关键点置信度 (N,) float32 [0,1]
        skeleton: 骨架连接定义
        point_radius: 关键点半径
        skeleton_thickness: 骨架线宽
        color_scheme: 颜色方案 (hsv/confidence/custom)

    Returns:
        绘制后的BGR图像
    """
    img = image.copy()
    h, w = img.shape[:2]

    # 生成颜色
    if color_scheme == 'hsv':
        colors = [tuple(map(int, hsv_to_bgr(i/len(keypoints), 0.8, 0.8)))
                for i in range(len(keypoints))]
    elif color_scheme == 'confidence' and confidences is not None:
        colors = [(0, int(255*c), 255-int(255*c)) for c in confidences]
    else:
        colors = [(0, 255, 0)] * len(keypoints)

    # 绘制骨架连接
    for (j1, j2) in skeleton:
        if j1 < len(keypoints) and j2 < len(keypoints):
            pt1 = tuple(map(int, keypoints[j1]))
            pt2 = tuple(map(int, keypoints[j2]))
            if 0 <= pt1[0] < w and 0 <= pt1[1] < h and 0 <= pt2[0] < w and 0 <= pt2[1] < h:
                cv2.line(img, pt1, pt2, (255, 255, 255), skeleton_thickness)

    # 绘制关键点
    for i, (x, y) in enumerate(keypoints):
        x, y = int(x), int(y)
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(img, (x, y), point_radius, colors[i], -1)

    return img

def heatmaps_to_keypoints(
    heatmaps: torch.Tensor,
    original_size: Tuple[int, int],
    threshold: float = 0.1,
    use_subpixel: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    高精度热图转关键点坐标，支持亚像素定位

    Args:
        heatmaps: 关键点热图 (C,H,W) float32
        original_size: 原始图像尺寸 (W,H)
        threshold: 置信度阈值
        use_subpixel: 是否启用亚像素精度

    Returns:
        (关键点坐标 (C,2), 置信度 (C,))
    """
    device = heatmaps.device
    C, H, W = heatmaps.shape
    heatmaps = heatmaps.cpu().numpy()

    keypoints = np.zeros((C, 2), dtype=np.float32)
    confidences = np.zeros(C, dtype=np.float32)
    target_w, target_h = original_size

    for i in range(C):
        hm = cv2.GaussianBlur(heatmaps[i], (3,3), 0)
        conf = hm.max()
        confidences[i] = conf

        if conf < threshold:
            continue

        # 亚像素定位
        if use_subpixel:
            y, x = np.unravel_index(hm.argmax(), hm.shape)
            if 0 < x < W-1 and 0 < y < H-1:
                dx = (hm[y, x+1] - hm[y, x-1]) / 2.0
                dy = (hm[y+1, x] - hm[y-1, x]) / 2.0
                dxx = hm[y, x+1] - 2*hm[y, x] + hm[y, x-1]
                dyy = hm[y+1, x] - 2*hm[y, x] + hm[y-1, x]

                if dxx != 0:
                    x = x - dx / dxx
                if dyy != 0:
                    y = y - dy / dyy

        # 缩放到原始图像尺寸
        keypoints[i] = (x * (target_w / W), y * (target_h / H))

    return keypoints, confidences

def hsv_to_bgr(h: float, s: float, v: float) -> np.ndarray:
    """HSV转BGR颜色"""
    return cv2.cvtColor(
        np.array([[[h*180, s*255, v*255]]], dtype=np.uint8),
        cv2.COLOR_HSV2BGR
    )[0][0]
