import os
import torch
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval
import wandb
from .network_utils import decode_simcc
from .dataset_util import draw_pose_on_image
from PIL import Image
import random


def evaluate(model, val_loader, device, input_size, bins, coco_gt, n_viz=16):
    model.eval()
    results = []
    viz_images = []

    input_w, input_h = input_size
    all_indices = list(range(len(val_loader)))
    viz_indices = set(random.sample(all_indices, min(n_viz, len(all_indices))))
    coco_gt = val_loader.dataset.coco

    with torch.no_grad():
        for i, (img_tensor, meta) in enumerate(val_loader):
            img_tensor = img_tensor.to(device)
            image_id = meta['image_id'].item()
            bbox = meta['bbox'].squeeze(0).tolist()
            pred_x, pred_y, _ = model(img_tensor)

            keypoints = decode_simcc(pred_x, pred_y, input_size, bins)  # [B, K, 2]
            kps = keypoints[0].cpu().numpy()  # [K,2]

            # Reverse affine mapping to original image coordinates
            x0, y0, w, h = bbox
            sx = input_w / w
            sy = input_h / h
            kps[:, 0] = kps[:, 0] / sx + x0
            kps[:, 1] = kps[:, 1] / sy + y0

            # Build COCO result entry
            keypoints_flat = []
            for (xx, yy) in kps:
                keypoints_flat.extend([float(xx), float(yy), 2.0])
            area = w * h
            img_id = int(meta['image_id'])
            result = {
                'image_id': img_id,
                'category_id': 1,
                'keypoints': keypoints_flat,
                'score': 1.0,
                'bbox': [x0, y0, w, h],
                'area': area
            }
            results.append(result)

            if i in viz_indices:
                file_name = val_loader.dataset.coco.loadImgs(image_id)[0]['file_name']
                img_path = os.path.join(val_loader.dataset.img_dir, file_name)
                orig_img = Image.open(img_path).convert('RGB')
                vis_img = draw_pose_on_image(orig_img, keypoints_flat)
                viz_images.append(wandb.Image(vis_img, caption=f"ID: {image_id}"))

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'keypoints')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mAP = float(coco_eval.stats[0])
    AP50 = float(coco_eval.stats[1])

    return mAP, AP50, viz_images
