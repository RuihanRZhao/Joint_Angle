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


def draw_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """
    在 RGB 图像上叠加单通道热力图。

    Args:
      - image: 原始 RGB 图像，shape=(H, W, 3)
      - heatmap: 单通道热力图，shape=(h, w)
      - alpha: 叠加权重，0~1，越大热力图越醒目

    Returns:
      - overlay: 叠加后的 RGB 图像，shape=(H, W, 3)
    """
    H, W = image.shape[:2]
    # 1) 先把 heatmap 归一化到 [0,255] 并转成 uint8
    hm = np.clip(heatmap, 0, 1)       # 确保在 [0,1]
    hm = (hm * 255).astype(np.uint8)  # [h, w]

    # 2) 上/下采样到和原图大小一致
    hm_resized = cv2.resize(hm, (W, H), interpolation=cv2.INTER_LINEAR)

    # 3) 应用伪彩色映射，得到 3 通道图
    hm_color = cv2.applyColorMap(hm_resized, cv2.COLORMAP_JET)  # [H, W, 3], BGR

    # 4) OpenCV 默认是 BGR，把原图也转为 BGR
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 5) 叠加
    overlay_bgr = cv2.addWeighted(img_bgr, 1-alpha, hm_color, alpha, 0)

    # 6) 再转回 RGB 返回
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    return overlay_rgb


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


def draw_keypoints_linked_multi(image, kps, vis, limb_pairs=None, color=(0,255,0), radius=3, thickness=2):
    """
    在 RGB 图像上绘制多人体关键点及连线。
    - image: numpy.ndarray，shape=(H,W,3)
    - kps:   可以是 list/Tensor/ndarray，表示 [(K,2),…] 或 已堆叠的 (P,K,2)
    - vis:   list/Tensor/ndarray，表示 [K,…] 或 (P,K)
    """
    # — 1. 统一格式到 numpy.ndarray —
    # 关键点
    if isinstance(kps, list):
        arrs = []
        for x in kps:
            if torch.is_tensor(x):
                arrs.append(x.cpu().numpy())
            else:
                arrs.append(np.asarray(x))
        kps = np.stack(arrs, axis=0)  # [P,K,2]
    elif torch.is_tensor(kps):
        kps = kps.cpu().numpy()
    else:
        kps = np.asarray(kps)

    # 可见性
    if isinstance(vis, list):
        arrs = []
        for x in vis:
            if torch.is_tensor(x):
                arrs.append(x.cpu().numpy())
            else:
                arrs.append(np.asarray(x))
        vis = np.stack(arrs, axis=0)  # [P,K] or if each vis is 1-D [K]
    elif torch.is_tensor(vis):
        vis = vis.cpu().numpy()
    else:
        vis = np.asarray(vis)

    # — 2. 兼容一维 vis（只给了每个人一个标志）—
    if vis.ndim == 1:
        P = kps.shape[0]
        K = kps.shape[1]
        # 把每个人所有关键点都当作可见
        vis = np.ones((P, K), dtype=int)

    # — 3. 准备骨架对列表 —
    P, K, _ = kps.shape
    H, W = image.shape[:2]
    if limb_pairs is None:
        limb_pairs = [
            (15,13),(13,11),(16,14),(14,12),(11,12),
            (5,11),(6,12),(5,6),(5,7),(6,8),
            (7,9),(8,10),(1,2),(0,1),(0,2),
            (1,3),(2,4),(3,5),(4,6)
        ]

    # — 4. 绘制 —
    output = image.copy()
    for p in range(P):
        for k in range(K):
            if vis[p, k] > 0:
                x, y = int(kps[p, k, 0]), int(kps[p, k, 1])
                cv2.circle(output, (x, y), radius, color, -1)
        for a, b in limb_pairs:
            if vis[p, a] > 0 and vis[p, b] > 0:
                xa, ya = int(kps[p, a, 0]), int(kps[p, a, 1])
                xb, yb = int(kps[p, b, 0]), int(kps[p, b, 1])
                cv2.line(output, (xa, ya), (xb, yb), color, thickness)

    return output


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
