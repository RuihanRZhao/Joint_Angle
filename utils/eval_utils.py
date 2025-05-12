import os
import random
import torch
import wandb
import cv2
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .dataset_util import visualization


def evaluate(model, val_loader, ann_file, val_image_dir, input_w, input_h, n_viz=5):
    """
    只使用 model 输出的 keypoints 分支进行 COCO 验证和可视化。

    Args:
        model: 已 load_state_dict 并 model.eval() 的网络，forward 返回 heatmap_init, heatmap_refine, keypoints。
        val_loader: 验证集 DataLoader，返回 (images, meta)。
        ann_file: 验证集 COCO keypoints JSON 路径。
        val_image_dir: 验证集图像目录。
        input_w, input_h: 模型输入裁剪图的宽高。
        n_viz: 随机可视化样本数。
    Returns:
        mAP (float), AP50 (float), List[wandb.Image]
    """
    was_train = model.training
    model.eval()
    device = next(model.parameters()).device

    # COCO ground truth
    coco_gt = val_loader.dataset.coco if hasattr(val_loader.dataset, 'coco') else COCO(
        os.path.join("run/single_person", ann_file))
    results = []
    viz_images = []

    total = len(val_loader.dataset)
    viz_idxs = set(random.sample(range(total), min(n_viz, total)))

    with torch.no_grad():
        for batch_idx, (images, meta) in tqdm(enumerate(val_loader), total=len(val_loader)):
            images = images.to(device)
            # Model forward
            heatmap_init, heatmap_refine, kpts = model(images)
            kpts = kpts.cpu().numpy()
            B = kpts.shape[0]

            for i in range(B):
                idx = batch_idx * val_loader.batch_size + i
                image_id = int(meta['image_id'][i])
                bbox = meta['bbox'][i].cpu().numpy()  # [x0, y0, w0, h0]
                x0, y0, w0, h0 = bbox

                # 模型输出的关键点坐标是归一化坐标 [-1,1]
                pts = kpts[i]  # shape [17,2] or [17,3]
                norm_xs = pts[:, 0]
                norm_ys = pts[:, 1]
                if pts.shape[1] == 3:
                    cs = pts[:, 2]
                else:
                    cs = np.ones(norm_xs.shape[0], dtype=np.float32)

                # 第一步：归一化 [-1,1] -> 输入图像像素坐标 📎
                px = (norm_xs + 1.0) / 2.0 * (input_w - 1)
                py = (norm_ys + 1.0) / 2.0 * (input_h - 1)

                # 计算原图像和目标输入的宽高比
                orig_ratio = w0 / h0 if h0 != 0 else 0.0
                target_ratio = input_w / input_h if input_h != 0 else 0.0

                # 第二步：输入图像像素坐标 -> 原图坐标 📎
                if abs(orig_ratio - target_ratio) < 1e-6:
                    # 无需 padding，直接线性缩放
                    xs = px * (w0 / (input_w - 1)) + x0
                    ys = py * (h0 / (input_h - 1)) + y0
                else:
                    # 使用了保持纵横比的缩放，计算统一的缩放比例
                    scale = min(input_w / w0, input_h / h0) if (w0 != 0 and h0 != 0) else 1.0
                    xs = px / scale + x0
                    ys = py / scale + y0

                # 将关键点坐标反归一化回原图尺度（并确保在图像边界内） 📎
                img_info = coco_gt.loadImgs(image_id)[0] if coco_gt else None
                if img_info is not None:
                    img_width, img_height = img_info['width'], img_info['height']
                    xs = np.clip(xs, 0, img_width - 1)
                    ys = np.clip(ys, 0, img_height - 1)

                # COCO 格式 keypoints 列表
                keypoints_list = []
                for x_pred, y_pred, c in zip(xs, ys, cs):
                    keypoints_list += [float(x_pred), float(y_pred), float(c)]
                # Use average of confidence scores (or 1.0 if not provided) as instance score
                score = float(np.mean(cs))

                results.append({
                    'image_id': image_id,
                    'category_id': 1,
                    'keypoints': keypoints_list,
                    'score': score
                })

                # 可视化
                if idx in viz_idxs:
                    # Draw keypoints on image for visualization
                    orig_img_path = os.path.join(val_image_dir, f"{image_id:012d}.jpg")
                    orig_img = cv2.imread(orig_img_path)
                    for x_pred, y_pred, conf in zip(xs, ys, cs):
                        if conf > 0.05:  # draw if confident
                            cv2.circle(orig_img, (int(x_pred), int(y_pred)), 3, (0, 0, 255), -1)
                    viz_images.append(wandb.Image(orig_img, caption=f"ID: {image_id}"))

    # Save results to JSON and evaluate with COCO API
    coco_dt = coco_gt.loadRes(results) if results else coco_gt
    coco_eval = COCOeval(coco_gt, coco_dt, 'keypoints')
    coco_eval.params.useSegm = False
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mAP, AP50 = float(coco_eval.stats[0]), float(coco_eval.stats[1])

    if was_train:
        model.train()
    return mAP, AP50, viz_images
