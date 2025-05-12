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
    适配 SimCC 坐标体系的评估函数（新版）：
    - 不再使用 bbox 映射坐标
    - 直接将预测从 input_size 缩放至原图尺寸
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
            img_id = int(meta['image_id'])
            img_tensor = img_tensor.to(device)

            # 模型前向
            pred_x, pred_y = model(img_tensor)
            coords, conf = decode_simcc(pred_x, pred_y, input_size, bins, return_score=True)

            # 取第一个样本
            kps = coords[0].cpu().numpy()  # [K, 2]
            scores = conf[0].cpu().numpy()  # [K]

            # 获取原图尺寸（从 COCO GT 信息）
            orig_img_info = coco_gt.loadImgs(img_id)[0]
            orig_w, orig_h = orig_img_info['width'], orig_img_info['height']

            # 坐标缩放回原图尺寸
            kps[:, 0] = kps[:, 0] / input_w * orig_w
            kps[:, 1] = kps[:, 1] / input_h * orig_h

            # 构建 COCO 格式关键点
            keypoints_flat = []
            for (x, y), s in zip(kps, scores):
                v = 2 if s >= conf_threshold else 1  # 2=visible, 1=low conf
                keypoints_flat.extend([float(x), float(y), v])

            result = {
                'image_id': img_id,
                'category_id': 1,
                'keypoints': keypoints_flat,
                'score': float(scores.mean()),
            }
            results.append(result)

            # 可视化（随机采样）
            if i in viz_idxs:
                orig_img_path = os.path.join(val_loader.dataset.img_dir, orig_img_info['file_name'])
                orig_img = Image.open(orig_img_path).convert('RGB')

                # 加载GT关键点
                ann_ids = coco_gt.getAnnIds(imgIds=img_id, catIds=[1])
                if not ann_ids:
                    continue
                gt_kps = coco_gt.loadAnns(ann_ids)[0]['keypoints']

                vis_img = draw_pose_on_image(orig_img.copy(), gt_kps, color=(0, 255, 0))  # GT绿色
                vis_img = draw_pose_on_image(vis_img, keypoints_flat, color=(255, 0, 0))  # Pred红色
                viz_images.append(wandb.Image(vis_img, caption=f"ID[{img_id}]"))

    # === COCO Evaluation ===
    if not results:
        print("No valid predictions generated.")
        return 0.0, 0.0, []

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'keypoints')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mAP = float(coco_eval.stats[0])
    AP50 = float(coco_eval.stats[1])

    return mAP, AP50, viz_images
