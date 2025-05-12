import os
import torch
import numpy as np
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval
from .network_utils.decoder import decode_simcc
from .dataset_util.visualization import draw_pose_on_image
from PIL import Image
import random
import wandb

def evaluate(model, val_loader, device, input_size, bins, n_viz=16, conf_thresh=0.3):
    """
    评估函数（改进版）：
    - 使用 softmax 解码 SimCC
    - 返回 AP/mAP 指标
    - 可视化样本（预测 vs GT）
    - 应用关键点置信度阈值过滤
    """
    model.eval()
    results = []
    viz_images = []

    input_w, input_h = input_size
    total = len(val_loader.dataset)
    coco_gt = val_loader.dataset.coco

    # 随机选 n_viz 张图可视化
    viz_idxs = set(random.sample(range(total), min(n_viz, total)))

    with torch.no_grad():
        for i, (img_tensor, meta) in enumerate(tqdm(val_loader, desc='Evaluating')):
            bbox = meta['bbox'].squeeze(0).tolist()
            img_id = int(meta['image_id'])
            img_tensor = img_tensor.to(device)

            # 模型前向
            pred_x, pred_y = model(img_tensor)
            keypoints, keypoint_scores = decode_simcc(pred_x, pred_y, input_size, bins, return_score=True)
            kps = keypoints[0].cpu().numpy()
            scores = keypoint_scores[0].cpu().numpy()

            # 映射回原图坐标系
            x0, y0, w, h = bbox
            sx = input_w / w
            sy = input_h / h
            kps[:, 0] = kps[:, 0] / sx + x0
            kps[:, 1] = kps[:, 1] / sy + y0

            # 构造 keypoints flat list
            keypoints_flat = []
            for (x, y, conf) in zip(kps[:, 0], kps[:, 1], scores):
                if conf < conf_thresh:
                    keypoints_flat.extend([float(x), float(y), 0.0])
                else:
                    keypoints_flat.extend([float(x), float(y), float(conf)])

            result = {
                'image_id': img_id,
                'category_id': 1,
                'keypoints': keypoints_flat,
                'score': float(scores.mean()),  # 平均置信度作为实例分数
                'bbox': [x0, y0, w, h],
                'area': w * h
            }
            results.append(result)

            # 可视化
            if i in viz_idxs:
                img_path = os.path.join(val_loader.dataset.img_dir,
                                        coco_gt.loadImgs(img_id)[0]['file_name'])
                orig_img = Image.open(img_path).convert('RGB')

                # GT 和预测关键点可视化
                gt_kps = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id, catIds=[1]))[0]['keypoints']
                vis_img = draw_pose_on_image(orig_img.copy(), gt_kps, color=(0, 255, 0))   # 绿色 GT
                vis_img = draw_pose_on_image(vis_img, keypoints_flat, color=(255, 0, 0))    # 红色 Pred
                viz_images.append(wandb.Image(vis_img, caption=f"ID[{img_id}]"))

    # COCO mAP/AP50
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'keypoints')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mAP = float(coco_eval.stats[0])
    AP50 = float(coco_eval.stats[1])

    return mAP, AP50, viz_images
