import os
import random
import torch
import wandb
import cv2
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .dataset_util import batch_heatmaps_to_keypoints, draw_pose_on_image

def evaluate(model, val_loader, ann_file, val_image_dir, n_viz=5):
    """
    使用 COCO 官方评估工具对模型进行验证，并可视化若干样本。

    Args:
        model: 已加载权重并设置为 eval 模式的网络模型。
        val_loader: 验证集 DataLoader，返回 (images, meta) 元组。
        ann_file: COCO 骨架关键点验证集标注 JSON 路径。
        val_image_dir: 验证集图像目录路径（val2017）。
        n_viz: 随机可视化图像数量。
    Returns:
        mAP (float), AP50 (float), List[wandb.Image]
    """
    was_train = model.training
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # Load COCO ground truth annotations
    coco_gt = val_loader.dataset.coco if hasattr(val_loader.dataset, 'coco') else COCO(ann_file)
    results = []
    viz_images = []
    # Pre-select random indices for visualization
    total_samples = len(val_loader.dataset)
    viz_idxs = set(random.sample(range(total_samples), min(n_viz, total_samples)))
    with torch.no_grad():
        for batch_idx, (images, meta) in enumerate(val_loader):
            images = images.to(device)
            outputs = model(images)
            heatmaps = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs
            preds = batch_heatmaps_to_keypoints(heatmaps)
            if isinstance(preds, torch.Tensor):
                preds = preds.cpu().numpy()
            B = preds.shape[0]
            # Process each sample in batch
            for i in range(B):
                global_idx = batch_idx * val_loader.batch_size + i
                image_id = int(meta['image_id'][i]) if isinstance(meta['image_id'], torch.Tensor) else int(meta['image_id'][i])
                # Retrieve original bounding box and image info
                bbox = meta['bbox'][i].cpu().numpy() if isinstance(meta['bbox'], torch.Tensor) else np.array(meta['bbox'][i])
                x0, y0, w0, h0 = bbox.tolist()
                img_info = coco_gt.loadImgs(image_id)[0]
                img_width, img_height = img_info['width'], img_info['height']
                # Decode predicted keypoints to original image coordinates
                kpts = preds[i]
                # Heatmap dimensions (from predictions shape)
                H_out, W_out = heatmaps.shape[-2], heatmaps.shape[-1]
                # Map normalized coordinates
                orig_kpts = np.zeros_like(kpts)
                orig_kpts[:,0] = x0 + (kpts[:,0] / W_out) * w0
                orig_kpts[:,1] = y0 + (kpts[:,1] / H_out) * h0
                orig_kpts[:,2] = kpts[:,2]
                # Prepare keypoints list in COCO format
                kpts_list = []
                for kp in orig_kpts:
                    kpts_list.extend([float(kp[0]), float(kp[1]), float(kp[2])])
                score = float(np.mean(orig_kpts[:,2]))
                results.append({
                    'image_id': image_id,
                    'category_id': 1,
                    'keypoints': kpts_list,
                    'score': score
                })
                # Visualization
                if global_idx in viz_idxs:
                    img_path = os.path.join(val_image_dir, img_info['file_name'])
                    orig_bgr = cv2.imread(img_path)
                    if orig_bgr is None:
                        continue
                    img_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
                    img_tensor = torch.from_numpy(img_rgb).permute(2,0,1).to(dtype=torch.uint8)
                    # Draw ground truth keypoints in green
                    ann_ids = coco_gt.getAnnIds(image_id, catIds=[1])
                    gt_ann = coco_gt.loadAnns(ann_ids)[0] if len(ann_ids) > 0 else None
                    if gt_ann:
                        gt_kps = np.array(gt_ann['keypoints'], dtype=np.float32).reshape(-1, 3)
                    else:
                        gt_kps = np.zeros((17,3), dtype=np.float32)
                    wb_gt = draw_pose_on_image(img_tensor, torch.from_numpy(gt_kps), color=(0,255,0))
                    base_img = torch.from_numpy(wb_gt.image).permute(2,0,1).to(dtype=torch.uint8)
                    pred_tensor = torch.from_numpy(orig_kpts)
                    wb_pred = draw_pose_on_image(base_img, pred_tensor, color=(255,0,0))
                    viz_images.append(wb_pred)


    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='keypoints')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    mAP, AP50 = float(coco_eval.stats[0]), float(coco_eval.stats[1])
    # Restore model training mode if it was in training
    if was_train:
        model.train()
    return mAP, AP50, viz_images
