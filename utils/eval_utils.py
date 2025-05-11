import os
import random
import torch
import wandb
import cv2
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .dataset_util import draw_pose_on_image

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

    # COCO GT
    coco_gt = val_loader.dataset.coco if hasattr(val_loader.dataset, 'coco') else COCO(os.path.join("run/single_person", ann_file))
    results = []
    viz_images = []

    total = len(val_loader.dataset)
    viz_idxs = set(random.sample(range(total), min(n_viz, total)))

    with torch.no_grad():
        for batch_idx, (images, meta) in tqdm(enumerate(val_loader),
                                             desc="Evaluating", total=len(val_loader),
                                             unit="batch"):
            images = images.to(device)
            # forward -> (heatmap_init, heatmap_refine, keypoints)
            _, _, kpts = model(images)
            # kpts: [B, 17, 2] or [B,17,3]
            kpts = kpts.cpu().numpy()
            B = kpts.shape[0]

            for i in range(B):
                idx = batch_idx * val_loader.batch_size + i
                image_id = int(meta['image_id'][i])
                # 原始 bbox
                bbox = meta['bbox'][i].cpu().numpy()  # [x0,y0,w0,h0]
                x0,y0,w0,h0 = bbox

                # 将 keypoints 从裁剪图坐标 映射回 原图坐标
                pts = kpts[i]  # shape [17,2] or [17,3]
                xs = pts[:,0] / input_w * w0 + x0
                ys = pts[:,1] / input_h * h0 + y0
                if pts.shape[1] == 3:
                    cs = pts[:,2]  # 网络输出的置信度
                else:
                    cs = np.ones(17, dtype=np.float32)  # 或取热图 max 作为置信度

                # COCO 格式 keypoints 列表
                keypoints_list = []
                for x,y,c in zip(xs, ys, cs):
                    keypoints_list += [float(x), float(y), float(c)]
                score = float(np.mean(cs))

                results.append({
                    'image_id': image_id,
                    'category_id': 1,
                    'keypoints': keypoints_list,
                    'score': score
                })

                # 可视化
                if idx in viz_idxs:
                    info = coco_gt.loadImgs(image_id)[0]
                    img_path = os.path.join(val_image_dir, info['file_name'])
                    img = cv2.imread(img_path)
                    if img is None: continue
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_tensor = torch.from_numpy(img_rgb).permute(2,0,1).to(device)

                    # GT keypoints
                    ann_ids = coco_gt.getAnnIds(image_id, catIds=[1])
                    gt_ann = coco_gt.loadAnns(ann_ids)[0] if ann_ids else None
                    if gt_ann:
                        gt_kps = np.array(gt_ann['keypoints'], dtype=np.float32).reshape(-1,3)
                    else:
                        gt_kps = np.zeros((17,3), dtype=np.float32)

                    # 在原图画 GT (绿) 和 Pred (红)
                    wb_gt, _ = draw_pose_on_image(img_tensor,
                                                  torch.from_numpy(gt_kps).to(device),
                                                  color=(0,255,0))
                    pred_kpts = np.stack([xs, ys, cs], axis=1)
                    _, wb_pred = draw_pose_on_image(wb_gt,
                                                   torch.from_numpy(pred_kpts).to(device),
                                                   color=(255,0,0),
                                                   use_wandb=True)
                    viz_images.append(wb_pred)

    # COCO 评估
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='keypoints')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    mAP, AP50 = float(coco_eval.stats[0]), float(coco_eval.stats[1])

    if was_train:
        model.train()
    return mAP, AP50, viz_images
