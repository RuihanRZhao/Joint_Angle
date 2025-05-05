import cv2
import numpy as np
from typing import List, Tuple, Dict

from pycocotools.coco import COCO


# COCO skeleton definition: list of (start_kp, end_kp) index pairs
COCO_PERSON_SKELETON: List[Tuple[int, int]] = [
    (15, 13), (13, 11), (16, 14), (14, 12),
    (5, 11), (6, 12), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1),
    (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)
]


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
    out_w, out_h = output_size

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

if __name__ == '__main__':
    """
    测试 visualize_coco_gt_keypoints 函数的主函数。
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="测试 COCO GT 关键点可视化"
    )
    parser.add_argument(
        '--img_path', type=str, default= '../run/data/val2017/000000000885.jpg',
        help='原始图像路径'
    )
    parser.add_argument(
        '--ann_file', type=str, default= '../run/data/annotations/person_keypoints_val2017.json',
        help='COCO 注释文件路径'
    )
    parser.add_argument(
        '--img_id', type=int, default= 885,
        help='COCO 图像 ID，例如 1'
    )
    parser.add_argument(
        '--out', type=str, default='vis_gt.jpg',
        help='可视化输出文件路径'
    )
    parser.add_argument(
        '--width', type=int, default=192,
        help='可视化图像宽度'
    )
    parser.add_argument(
        '--height', type=int, default=256,
        help='可视化图像高度'
    )
    args = parser.parse_args()

    # 加载原始图像
    img = cv2.imread(args.img_path)
    if img is None:
        print(f"无法加载图像: {args.img_path}")

    # 加载 COCO 注释
    coco = COCO(args.ann_file)
    anns = coco.loadAnns(coco.getAnnIds(
        imgIds=[args.img_id], catIds=[1], iscrowd=None
    ))

    # 调用可视化函数
    vis_img = visualize_coco_keypoints(
        img, anns, COCO_PERSON_SKELETON,
        (args.width, args.height),
        (255,0,0),
        (255,0,0)
    )

    # 显示并保存结果
    cv2.imshow('GT Keypoints', vis_img)
    cv2.imwrite(args.out, vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()