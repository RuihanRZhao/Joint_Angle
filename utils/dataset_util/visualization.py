import numpy as np
import cv2
from PIL import Image

# COCO 17 keypoints skeleton connections
COCO_PERSON_SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (0,5),(0,6),(5,7),(7,9),
    (6,8),(8,10),(5,6),(11,12),
    (5,11),(6,12),(11,13),(13,15),
    (12,14),(14,16)
]

def draw_pose_on_image(img, keypoints_list, color=(255, 0, 0)):
    """
    img: np.ndarray or PIL.Image, shape (H, W, 3), dtype=uint8
    keypoints_list: list or np.ndarray of shape [3*K], format: [x1, y1, v1, ..., xK, yK, vK]
    color: tuple, RGB color
    returns: np.ndarray
    """
    if isinstance(img, Image.Image):
        img = np.array(img)

    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    img_draw = img.copy()
    kp = np.array(keypoints_list).reshape(-1, 3)  # [K, 3]

    # COCO 骨架连接（部分示例）
    skeleton = COCO_PERSON_SKELETON

    # 画关键点
    for i, (x, y, v) in enumerate(kp):
        if v > 0:
            cv2.circle(img_draw, (int(x), int(y)), 3, color, -1)

    # 画骨架线段
    for i, j in skeleton:
        if kp[i, 2] > 0 and kp[j, 2] > 0:
            pt1 = (int(kp[i, 0]), int(kp[i, 1]))
            pt2 = (int(kp[j, 0]), int(kp[j, 1]))
            cv2.line(img_draw, pt1, pt2, color, 2)

    img_draw = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)

    return img_draw
