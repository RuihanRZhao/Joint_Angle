import os
import random
import torch
import wandb
import cv2
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .dataset_util.visualization import draw_pose_on_image

def evaluate(model, val_loader, ann_file, val_image_dir, input_w, input_h, n_viz=5):
    """
    使用 COCO keypoints 模式评估模型并可视化 GT 与预测。

    Args:
        model: 已加载并设置为 eval 模式的网络，forward 返回 heatmap_init, heatmap_refine, keypoints。
        val_loader: 验证集 DataLoader，返回 (images, meta)。
        ann_file: COCO 骨架关键点注解 JSON 文件路径。
        val_image_dir: 验证集原始图像目录路径。
        input_w, input_h: 模型输入裁剪图像宽度和高度。
        n_viz: 随机可视化样本数量。

    Returns:
        mAP (float), AP50 (float), List[wandb.Image]
    """
    was_train = model.training
    model.eval()
    device = next(model.parameters()).device

    # 加载 COCO GT
    coco_gt = val_loader.dataset.coco if hasattr(val_loader.dataset, 'coco') else COCO(ann_file)
    results = []
    viz_images = []

    total = len(val_loader.dataset)
    viz_idxs = set(random.sample(range(total), min(n_viz, total)))

    with torch.no_grad():
        for batch_idx, (images, meta) in tqdm(enumerate(val_loader), total=len(val_loader)):
            images = images.to(device)
            heatmap_init, heatmap_refine, kpts = model(images)
            kpts = kpts.cpu().numpy()
            B = kpts.shape[0]

            for i in range(B):
                idx = batch_idx * val_loader.batch_size + i
                image_id = int(meta['image_id'][i])
                bbox = meta['bbox'][i].cpu().numpy()  # [x0, y0, w0, h0]
                x0, y0, w0, h0 = bbox

                # 归一化坐标 -> 输入像素坐标
                pts = kpts[i]  # shape [J,2]
                norm_xs = pts[:, 0]
                norm_ys = pts[:, 1]
                cs = np.ones_like(norm_xs, dtype=np.float32)
                px = (norm_xs + 1.0) / 2.0 * (input_w - 1)
                py = (norm_ys + 1.0) / 2.0 * (input_h - 1)

                # 输入像素 -> 原图坐标
                orig_info = coco_gt.loadImgs(image_id)[0]
                orig_w, orig_h = orig_info['width'], orig_info['height']
                orig_ratio = w0 / h0 if h0 > 0 else 0.0
                target_ratio = input_w / input_h if input_h > 0 else 0.0
                if abs(orig_ratio - target_ratio) < 1e-6:
                    xs = px * (w0 / (input_w - 1)) + x0
                    ys = py * (h0 / (input_h - 1)) + y0
                else:
                    scale = min(input_w / w0, input_h / h0) if (w0 > 0 and h0 > 0) else 1.0
                    xs = px / scale + x0
                    ys = py / scale + y0
                xs = np.clip(xs, 0, orig_w - 1)
                ys = np.clip(ys, 0, orig_h - 1)

                # 构造 COCO dt 格式 keypoints list
                dt_keypoints = []
                for x_pred, y_pred, c in zip(xs, ys, cs):
                    dt_keypoints += [float(x_pred), float(y_pred), float(c)]
                score = float(np.mean(cs))
                results.append({'image_id': image_id, 'category_id': 1,
                                'keypoints': dt_keypoints, 'score': score})

                # 可视化 GT & DT
                if idx in viz_idxs:
                    # 读取原图
                    img_file = orig_info['file_name']
                    orig_path = os.path.join(val_image_dir, img_file)
                    orig_img = cv2.imread(orig_path)
                    if orig_img is None:
                        raise RuntimeError(f"无法读取可视化图像：{orig_path}")
                    # 获取 GT keypoints
                    ann_ids = coco_gt.getAnnIds(imgIds=image_id, catIds=[1])
                    gt_keypoints = coco_gt.loadAnns(ann_ids)[0]['keypoints']
                    # 在同一张图上先绘制 GT（绿色），再绘制 DT（红色）
                    vis = draw_pose_on_image(orig_img, gt_keypoints, color=(0,255,0))
                    vis = draw_pose_on_image(vis, dt_keypoints, color=(0,0,255))
                    viz_images.append(wandb.Image(vis, caption=f"ID:{image_id}"))

    # COCOeval
    coco_dt = coco_gt.loadRes(results)

    print(coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=image_id, catIds=[1])), coco_dt.loadAnns(coco_gt.getAnnIds(imgIds=image_id, catIds=[1])))

    coco_eval = COCOeval(coco_gt, coco_dt, 'keypoints')

    # —— 强制覆盖 ——
    coco_eval.params.iouType = 'keypoints'  # 一定要是 'keypoints'
    coco_eval.params.maxDets = [20, 50, 100]  # keypoints 默认的 maxDets
    if hasattr(coco_eval.params, 'useSegm'):
        coco_eval.params.useSegm = None  # 清掉这个参数，避免触发 bbox 或 segm 分支
    print(
        f"[DEBUG] iouType={coco_eval.params.iouType}, maxDets={coco_eval.params.maxDets}, useSegm={coco_eval.params.useSegm}")
    # —— 覆盖结束 ——

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mAP, AP50 = float(coco_eval.stats[0]), float(coco_eval.stats[1])
    if was_train:
        model.train()
    return mAP, AP50, viz_images
