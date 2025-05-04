import cv2
import numpy as np
import torch

# COCO 骨骼对 (0-based index)
SKELETON = [
    (15,13),(13,11),(16,14),(14,12),(11,12),
    (5,11),(6,12),(5,6),(5,7),(7,9),
    (6,8),(8,10),(1,2),(0,1),(0,2),
    (1,3),(2,4),(3,5),(4,6)
]


def overlay_mask(image, mask, alpha=0.5, color=(0,255,0)):
    """
    将二值 mask 叠加到 image 上。
    image: BGR uint8 HxWx3
    mask:  [H, W] 0/1 uint8 or bool
    """
    m = (mask > 0).astype(np.uint8) * 255
    m_bgr = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(image, 1-alpha, m_bgr, alpha, 0)
    return overlay


def draw_heatmap(image, heatmap, alpha=0.6):
    """
    将单通道 heatmap 正规化后叠加到 image 上。
    heatmap: HxW float32
    """
    hm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 1-alpha, hm_color, alpha, 0)
    return overlay


def draw_paf(image, paf, stride=8, scale=4, color=(255,0,0), thickness=1):
    """
    在 image 上绘制 PAF 向量场。
    paf: [2L, H, W] float32, x-通道，y-通道交替存储
    stride: 每隔 stride 像素绘制一个向量
    scale: 缩放向量长度显示
    """
    h, w = paf.shape[1:]
    out = image.copy()
    for idx, (a,b) in enumerate(SKELETON):
        vx = paf[2*idx]
        vy = paf[2*idx+1]
        # 在网格点上绘制
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                pt1 = (int(x*scale), int(y*scale))
                dx = int(vx[y,x] * scale)
                dy = int(vy[y,x] * scale)
                pt2 = (pt1[0]+dx, pt1[1]+dy)
                cv2.arrowedLine(out, pt1, pt2, color, thickness, tipLength=0.3)
    return out


def draw_keypoints_linked_multi(
    image,
    kps,
    vis,
    kp_color=(0,255,0),
    line_color=(0,255,255),
    radius=3,
    kp_thickness=-1,
    line_thickness=2
):
    """
    在 image 上绘制多人体关键点和骨骼连线。
    kps: np.ndarray 或 Tensor, [P, 17, 2]
    vis: np.ndarray 或 Tensor, [P, 17]
    """
    if torch.is_tensor(kps):
        kps = kps.cpu().numpy()
    if torch.is_tensor(vis):
        vis = vis.cpu().numpy()
    out = image.copy()
    P, K, _ = kps.shape
    for p in range(P):
        person_kps = kps[p]
        person_vis = vis[p]
        # draw lines
        for a,b in SKELETON:
            if person_vis[a] > 0 and person_vis[b] > 0:
                x1,y1 = person_kps[a]
                x2,y2 = person_kps[b]
                pt1 = (int(round(x1)), int(round(y1)))
                pt2 = (int(round(x2)), int(round(y2)))
                cv2.line(out, pt1, pt2, line_color, line_thickness)
        # draw keypoints
        for (x,y), v in zip(person_kps, person_vis):
            if v > 0:
                cv2.circle(out, (int(round(x)), int(round(y))), radius, kp_color, kp_thickness)
    return out


def visualize_sample(
    image,
    mask=None,
    heatmaps=None,
    pafs=None,
    multi_kps=None,
    multi_vis=None
):
    """
    综合可视化：
      - mask: segmentation overlay
      - heatmaps: list 或 ndarray [K,H,W]
      - pafs: ndarray [2L,H,W]
      - multi_kps/multi_vis: [P,17,2] & [P,17]
    返回 dict of {name: image}
    """
    outputs = {}
    base = image.copy()
    if mask is not None:
        outputs['segmentation'] = overlay_mask(base, mask)
    if heatmaps is not None:
        # 展示第一个 heatmap 作为示例，可自行选通道
        hm0 = heatmaps[0]
        outputs['heatmap_0'] = draw_heatmap(base, hm0)
    if pafs is not None:
        outputs['paf_field'] = draw_paf(base, pafs)
    if multi_kps is not None and multi_vis is not None:
        outputs['keypoints'] = draw_keypoints_linked_multi(base, multi_kps, multi_vis)
    return outputs
