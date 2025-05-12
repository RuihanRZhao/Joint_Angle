import os
import torch
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval
import wandb
from .network_utils import decode_simcc
from .dataset_util import draw_pose_on_image
from PIL import Image
import random

def evaluate(model, val_loader, device, input_size, bins, n_viz=16):
    model.eval()
    results = []
    viz_images = []
    coco_gt = val_loader.dataset.coco

    input_w, input_h = input_size
    total = len(val_loader.dataset)
    viz_idxs = set(random.sample(range(total), min(n_viz, total)))
    with torch.no_grad():
        for i, (img_tensor, meta) in enumerate(tqdm(val_loader, desc='Evaluating')):
            bbox = meta['bbox'].squeeze(0).tolist()
            img_tensor =img_tensor.to(device)
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

            # 可视化图像
            if i in viz_idxs:
                orig_img = Image.open(os.path.join(val_loader.dataset.img_dir, val_loader.dataset.coco.loadImgs(img_id)[0]['file_name'])).convert('RGB')
                vis_img = draw_pose_on_image(orig_img.copy(), coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id, catIds=[1]))[0]['keypoints'],(0, 255, 0))
                vis_img = draw_pose_on_image(vis_img, keypoints_flat, (0,0,255))
                viz_images.append(wandb.Image(vis_img, caption=f"ID[{img_id}]"))

    # 调用 COCOeval
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'keypoints')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mAP = float(coco_eval.stats[0])
    AP50 = float(coco_eval.stats[1])

    return mAP, AP50, viz_images
