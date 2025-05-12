import os
import torch
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval
import wandb
from .network_utils import decode_simcc
from .dataset_util import draw_pose_on_image
from PIL import Image


def evaluate(model, val_loader, device, input_size, bins, n_viz=16):
    model.eval()
    results = []
    viz_images = []
    coco_gt = val_loader.dataset.coco

    with torch.no_grad():
        for i, (img_tensor, meta) in enumerate(tqdm(val_loader, desc='Evaluating')):
            img_tensor = img_tensor.to(device)
            image_id = meta['image_id'].item()
            bbox = meta['bbox'].squeeze(0).tolist()  # [x,y,w,h]
            area = bbox[2] * bbox[3]

            # Forward pass
            pred_x, pred_y, _ = model(img_tensor)  # add batch dim
            coords = decode_simcc(pred_x, pred_y, input_size=input_size, bins=bins)  # [1, K, 2]
            keypoints = coords[0].cpu().numpy()

            # 生成 COCO keypoints 格式 (x1,y1,v1,...,xJ,yJ,vJ)
            keypoints_flat = []
            for (x, y) in keypoints:
                keypoints_flat.extend([float(x), float(y), 2.0])

            results.append({
                'image_id': int(image_id),
                'category_id': 1,
                'keypoints': keypoints_flat,
                'score': 1.0,
                'area': float(area),
                'bbox': bbox
            })

            # 可视化图像
            if i < n_viz:
                orig_img = Image.open(os.path.join(val_loader.dataset.img_dir, val_loader.dataset.coco.loadImgs(image_id)[0]['file_name'])).convert('RGB')
                vis_img = draw_pose_on_image(orig_img.copy(), coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=image_id, catIds=[1]))[0]['keypoints'],(0, 255, 0))
                vis_img = draw_pose_on_image(vis_img, keypoints_flat, (0,0,255))
                viz_images.append(wandb.Image(vis_img, caption=f"ID[{image_id}]"))

    # 调用 COCOeval
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'keypoints')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mAP = float(coco_eval.stats[0])
    AP50 = float(coco_eval.stats[1])

    return mAP, AP50, viz_images
