import os
import torch
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval
from .network_utils.decoder import decode_simcc
from .dataset_util.visualization import draw_pose_on_image
from PIL import Image
import random
import wandb

def evaluate(model, val_loader, device, input_size, bins, n_viz=16, conf_threshold=0.0):
    """
    重构后的评估函数，支持 soft-argmax + score 筛除
    """
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
            img_id = int(meta['image_id'])

            img_tensor = img_tensor.to(device)
            pred_x, pred_y = model(img_tensor)

            coords, conf = decode_simcc(pred_x, pred_y, input_size, bins, return_score=True)
            kps = coords[0].cpu().numpy()  # [K, 2]
            scores = conf[0].cpu().numpy()  # [K]

            # 反映射坐标（归一化 bbox → 原图）
            x0, y0, w, h = bbox
            sx = input_w / w
            sy = input_h / h
            kps[:, 0] = kps[:, 0] / sx + x0
            kps[:, 1] = kps[:, 1] / sy + y0

            # 关键点输出：[x, y, v]，v 置信度
            keypoints_flat = []
            for (x, y), s in zip(kps, scores):
                v = 2 if s >= conf_threshold else 1  # 2: visible, 1: low conf
                keypoints_flat.extend([float(x), float(y), v])

            result = {
                'image_id': img_id,
                'category_id': 1,
                'keypoints': keypoints_flat,
                'score': float(scores.mean()),
                'bbox': [x0, y0, w, h],
                'area': w * h
            }
            results.append(result)

            # 可视化
            if i in viz_idxs:
                orig_img = Image.open(os.path.join(
                    val_loader.dataset.img_dir,
                    val_loader.dataset.coco.loadImgs(img_id)[0]['file_name'])
                ).convert('RGB')

                gt_ann = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id, catIds=[1]))[0]['keypoints']
                vis_img = draw_pose_on_image(orig_img.copy(), gt_ann, color=(0, 255, 0))  # GT 绿色
                vis_img = draw_pose_on_image(vis_img, keypoints_flat, color=(255, 0, 0))  # Pred 红色

                viz_images.append(wandb.Image(vis_img, caption=f"ID[{img_id}]"))

    # COCO Eval
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'keypoints')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mAP = float(coco_eval.stats[0])
    AP50 = float(coco_eval.stats[1])

    return mAP, AP50, viz_images
