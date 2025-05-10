import cv2
import numpy as np

# COCO 17 点骨架连接关系（索引基于 COCO keypoints 顺序）
COCO_SKELETON = [
    (0,1), (0,2), (1,3), (2,4),
    (5,6), (5,7), (7,9), (6,8), (8,10),
    (5,11), (6,12), (11,12),
    (11,13), (13,15), (12,14), (14,16)
]

def draw_pose_on_image(
    image: np.ndarray,
    pred_kps: np.ndarray,
    gt_kps: np.ndarray,
    pred_scores: np.ndarray = None,
    gt_scores: np.ndarray = None
) -> np.ndarray:
    """
    在同一张图像上叠加绘制预测和真值关键点骨架。
    Args:
        image: 原始 BGR 图像 (H, W, 3)，uint8。
        pred_kps: 预测关键点数组 (17, 2)，每行 (x, y)。
        gt_kps: 真值关键点数组 (17, 2)，每行 (x, y)。
        pred_scores: 预测关键点置信度 (17,) 或 None。
        gt_scores: 真值关键点可见性 v (17,) 或 None。
    Returns:
        drawn: 叠加完成后的图像 (H, W, 3)。
    """
    drawn = image.copy()
    # 颜色定义：真值红色，预测绿色
    color_gt = (0, 0, 255)    # BGR 红
    color_pred = (0, 255, 0)  # BGR 绿

    # 绘制关键点函数
    def _draw_points(kps, scores, color):
        for i, (x, y) in enumerate(kps):
            if scores is None or scores[i] > 0:
                cv2.circle(drawn, (int(x), int(y)), 3, color, -1)

    # 绘制连线函数
    def _draw_skeleton(kps, scores, color):
        for i, j in COCO_SKELETON:
            xi, yi = kps[i]
            xj, yj = kps[j]
            if (scores is None or (scores[i] > 0 and scores[j] > 0)):
                cv2.line(drawn,
                         (int(xi), int(yi)),
                         (int(xj), int(yj)),
                         color, 2)

    # 真值
    _draw_skeleton(gt_kps, gt_scores, color_gt)
    _draw_points(  gt_kps, gt_scores, color_gt)
    # 预测
    _draw_skeleton(pred_kps, pred_scores, color_pred)
    _draw_points(  pred_kps, pred_scores, color_pred)

    return drawn