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
    model.eval()
    results = []
    viz_images = []

    coco_gt = val_loader.dataset.coco
    input_w, input_h = input_size
    total = len(val_loader.dataset)
    viz_idxs = set(random.sample(range(total), min(n_viz, total)))

    with torch.no_grad():
        for i, (img_tensor, meta) in enumerate(tqdm(val_loader, desc='Evaluating')):
            try:
                img_id = int(meta['image_id'])
                img_tensor = img_tensor.to(device)

                # Forward pass
                pred_x, pred_y = model(img_tensor)
                print(f"[{i}] pred_x.shape: {pred_x.shape}, pred_y.shape: {pred_y.shape}")

                coords, conf = decode_simcc(pred_x, pred_y, input_size, bins, return_score=True)
                print(f"[{i}] decoded coords shape: {coords.shape}, conf shape: {conf.shape}")

                kps = coords[0].cpu().numpy()
                scores = conf[0].cpu().numpy()

                # 原图尺寸
                img_info = coco_gt.loadImgs(img_id)[0]
                orig_w, orig_h = img_info['width'], img_info['height']
                print(f"[{i}] Original image size: ({orig_w}, {orig_h})")

                # 还原坐标
                kps[:, 0] = kps[:, 0] / input_w * orig_w
                kps[:, 1] = kps[:, 1] / input_h * orig_h

                # 关键点展平
                keypoints_flat = []
                for (x, y), s in zip(kps, scores):
                    v = 2 if s >= conf_threshold else 1
                    keypoints_flat.extend([float(x), float(y), v])

                print(f"[{i}] Sample keypoint: {keypoints_flat[:6]}")

                result = {
                    'image_id': img_id,
                    'category_id': 1,
                    'keypoints': keypoints_flat,
                    'score': float(scores.mean())
                }

                if all(p == 0 for p in keypoints_flat):
                    print(f"[{i}] ⚠️ Warning: All keypoints zero → skipping")
                    continue

                results.append(result)

                # 可视化
                if i in viz_idxs:
                    img_path = os.path.join(val_loader.dataset.img_dir, img_info['file_name'])
                    if not os.path.exists(img_path):
                        print(f"[{i}] ⚠️ Image path not found: {img_path}")
                        continue

                    orig_img = Image.open(img_path).convert("RGB")
                    ann_ids = coco_gt.getAnnIds(imgIds=img_id, catIds=[1])
                    if not ann_ids:
                        print(f"[{i}] ⚠️ No GT ann found for image_id={img_id}")
                        continue

                    gt_kps = coco_gt.loadAnns(ann_ids)[0]['keypoints']
                    vis_img = draw_pose_on_image(orig_img.copy(), gt_kps, color=(0, 255, 0))
                    vis_img = draw_pose_on_image(vis_img, keypoints_flat, color=(255, 0, 0))
                    viz_images.append(wandb.Image(vis_img, caption=f"ID[{img_id}]"))

            except Exception as e:
                print(f"[{i}] ❌ Exception during evaluation: {e}")
                continue

    print(f"\n✅ Total predictions generated: {len(results)}")
    if not results:
        print("❌ No valid predictions. Evaluation skipped.")
        return 0.0, 0.0, []

    # COCO Eval
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'keypoints')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mAP = float(coco_eval.stats[0])
    AP50 = float(coco_eval.stats[1])

    return mAP, AP50, viz_images
