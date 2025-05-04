import cv2
import os
import numpy as np
from PIL import Image, ImageDraw

def visualize_raw_samples(samples, save_dir):
    """
    可视化原始训练样本：叠加分割掩码和关键点后保存图像。

    参数:
        samples (List[Dict]): 由 prepare_coco_dataset 返回的样本列表
        save_dir (str): 可视化图像保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    for sample in samples:
        img = cv2.imread(sample["image_path"])
        if img is None:
            continue  # 图片读取失败则跳过
        mask = sample["mask"]
        keypoints = sample["keypoints"]
        visibility = sample["visibility"]

        # 将掩码区域设置为红色半透明
        red_mask = np.zeros_like(img)
        red_mask[:, :, 2] = (mask * 255).astype(np.uint8)  # OpenCV BGR 格式，红色通道
        overlay = cv2.addWeighted(img, 1.0, red_mask, 0.5, 0)

        # 绘制关键点为绿色小圆点
        for (x, y), vis in zip(keypoints, visibility):
            if vis > 0:
                center = (int(x), int(y))
                cv2.circle(overlay, center, radius=3, color=(0, 255, 0), thickness=-1)

        # 保存可视化图像
        filename = os.path.basename(sample["image_path"])
        cv2.imwrite(os.path.join(save_dir, filename), overlay)


def visualize_predictions(predictions, save_dir):
    """
    可视化模型预测结果，并可选叠加 GT 进行对比。

    参数:
        predictions (List[Dict]): 包含模型预测的列表，每个字典应至少包含:
            - image_path (str): 图像路径
            - mask (np.ndarray): 预测分割掩码 (H, W)
            - keypoints (np.ndarray): 预测关键点 (M,2)
            - visibility (np.ndarray): 预测关键点可见性 (M,)
            可选:
            - gt_mask (np.ndarray): GT 分割掩码
            - gt_keypoints (np.ndarray): GT 关键点 (N,2)
            - gt_visibility (np.ndarray): GT 可见性 (N,)
        save_dir (str): 可视化图像保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    for pred in predictions:
        img = cv2.imread(pred["image_path"])
        if img is None:
            continue
        overlay = img.copy().astype(np.float32)

        # 叠加预测掩码（红色）
        mask_pred = pred.get("mask")
        if mask_pred is not None:
            red_mask = np.zeros_like(img)
            red_mask[:, :, 2] = (mask_pred * 255).astype(np.uint8)
            overlay = cv2.addWeighted(overlay.astype(np.uint8), 1.0, red_mask, 0.5, 0)

        # 叠加 GT 掩码（绿色）
        mask_gt = pred.get("gt_mask")
        if mask_gt is not None:
            green_mask = np.zeros_like(img)
            green_mask[:, :, 1] = (mask_gt * 255).astype(np.uint8)
            overlay = cv2.addWeighted(overlay.astype(np.uint8), 1.0, green_mask, 0.5, 0)

        # 在图像上绘制预测关键点（红点）
        keypoints_pred = pred.get("keypoints", [])
        vis_pred = pred.get("visibility", [])
        for (x, y), v in zip(keypoints_pred, vis_pred):
            if v > 0:
                cv2.circle(overlay, (int(x), int(y)), 3, (0, 0, 255), -1)
        # 绘制 GT 关键点（绿点）
        keypoints_gt = pred.get("gt_keypoints", [])
        vis_gt = pred.get("gt_visibility", [])
        for (x, y), v in zip(keypoints_gt, vis_gt):
            if v > 0:
                cv2.circle(overlay, (int(x), int(y)), 3, (0, 255, 0), -1)

        # 保存可视化结果
        filename = os.path.basename(pred["image_path"])
        cv2.imwrite(os.path.join(save_dir, filename), overlay.astype(np.uint8))


COCO_SKELETON = [
    (0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),
    (6,8),(8,10),(11,12),(5,11),(6,12),(11,13),
    (13,15),(12,14),(14,16)
]

def draw_keypoints_linked(img_arr, kps, vis):
    """
    Draw keypoints and skeleton on image.
    img_arr: H×W×3 numpy img
    kps: [K,2] pixel coords
    vis: [K] visibility flags
    """
    vis_img = Image.fromarray(img_arr).convert('RGBA')
    draw = ImageDraw.Draw(vis_img)
    # points
    for (x,y), v in zip(kps, vis):
        if v > 0:
            draw.ellipse((x-3,y-3,x+3,y+3), fill=(255,0,0,180))
    # skeleton
    for i,j in COCO_SKELETON:
        if vis[i] > 0 and vis[j] > 0:
            x1,y1 = kps[i]; x2,y2 = kps[j]
            draw.line((x1,y1,x2,y2), fill=(0,255,0,180), width=2)
    return np.array(vis_img.convert('RGB'))