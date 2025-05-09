import cv2
import numpy as np
from typing import List, Tuple, Dict

from pycocotools.coco import COCO

from .coco import COCO_PERSON_SKELETON

def visualize_coco_keypoints(
    img: np.ndarray,
    anns: List[Dict],
    skeleton: List[Tuple[int, int]],
    output_size: Tuple[int, int],
    point_color: Tuple[int, int, int],
    line_color: Tuple[int, int, int],
) -> np.ndarray:
    """
    在原始图片上绘制 COCO 格式的 Ground Truth 关键点骨架。

    Args:
        img (np.ndarray): 通过 cv2.imread 读取的原始图像 (BGR 通道)，形状 (H, W, 3)。
        anns (List[Dict]): 来自 coco_gt.loadAnns 的注释列表，每个 ann 格式:
            {
                'keypoints': [x0, y0, v0, x1, y1, v1, ..., x16, y16, v16],  # 长度 = 3 * K（K=17）
                'num_keypoints': int,  # 标注的有效关键点数量
                ... # 其他字段可忽略
            }
        skeleton (List[Tuple[int,int]]): COCO 骨架拓扑结构，关键点索引对列表。
        output_size (Tuple[int,int]): 可视化时将图像缩放到的目标尺寸 (width, height)。

    Returns:
        vis_img (np.ndarray): 绘制好 GT 骨架的图，BGR 通道，形状 (output_h, output_w, 3)。
    """
    # 保存原始尺寸
    orig_h, orig_w = img.shape[:2]
    out_h, out_w = output_size

    # Resize 原图到可视化尺寸
    vis_img = cv2.resize(img, (out_w, out_h))

    # 绘制每个人的关键点和骨架连接
    for ann in anns:
        if ann.get('num_keypoints', 0) == 0:
            continue
        kps = ann['keypoints']  # 长度 3*K
        pts: List[Tuple[int,int]] = []
        # 按比例将 COCO (x,y) 映射到可视化图像尺寸
        for i in range(len(kps) // 3):
            x, y, v = kps[3*i:3*i+3]
            if v > 0:
                px = int(x * (out_w / orig_w))
                py = int(y * (out_h / orig_h))
                pts.append((px, py))
            else:
                pts.append(None)

        # 绘制骨架连线 (绿色)
        for a, b in skeleton:
            if pts[a] is not None and pts[b] is not None:
                cv2.line(vis_img, pts[a], pts[b], line_color, thickness=2)

        # 绘制关键点 (绿色实心圆)
        for pt in pts:
            if pt is not None:
                cv2.circle(vis_img, pt, radius=3, color=point_color, thickness=-1)

    return vis_img
