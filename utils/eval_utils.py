import os
import random
import torch
import wandb
import cv2
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils import batch_heatmaps_to_keypoints, draw_pose_on_image

def evaluate(model, val_loader, ann_file, val_image_dir, n_viz=5):
    """
    使用 COCO 官方评估工具对模型进行验证，并可视化若干样本。

    Args:
        model: 已加载权重并设置为 eval 模式的网络模型。
        val_loader: 验证集 DataLoader，返回 (images, meta) 元组。
        ann_file: COCO 骨架关键点验证集标注 JSON 路径。
        val_image_dir: 验证集图像目录路径（val2017）。
        n_viz: 随机可视化图像数量。
        post_process: 解码方式，仅支持 'gaussian'（由于 encoder/decoder 已集成亚像素修正）。

    Returns:
        mAP (float), AP50 (float), List[wandb.Image]
    """
    # 切换模型到评估模式
    was_train = model.training
    model.eval()

    # 加载 COCO 验证集标注
    coco_gt = COCO(val_loader.dataset.coco)

    results = []
    viz_images = []
    # 预选随机可视化索引
    total_samples = len(val_loader.dataset)
    viz_idxs = set(random.sample(range(total_samples), n_viz))

    device = next(model.parameters()).device

    with torch.no_grad():
        for batch_idx, (images, meta) in enumerate(val_loader):
            images = images.to(device)
            # 模型前向
            outputs = model(images)
            heatmaps = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs
            # 解码关键点预测
            preds = batch_heatmaps_to_keypoints(heatmaps)
            preds = preds

            B = preds.shape[0]
            for i in range(B):
                idx = batch_idx * val_loader.batch_size + i
                image_id = int(meta['image_id'][i])
                # 构建评估字典
                kpts = preds[i]
                kpts_list = kpts.view(-1).tolist()
                score = float(kpts[:,2].mean())
                results.append({
                    "image_id": image_id,
                    "category_id": 1,
                    "keypoints": kpts_list,
                    "score": score
                })
                # 可视化
                if idx in viz_idxs:
                    image_id = int(meta['image_id'][i])
                    info = coco_gt.loadImgs(image_id)[0]
                    img_path = os.path.join(val_image_dir, info['file_name'])
                    orig_bgr = cv2.imread(img_path)  # HxWx3, uint8, BGR

                    # 2) 获取 COCO 真值关键点
                    ann_ids = coco_gt.getAnnIds(image_id, catIds=[1])
                    gt_ann = coco_gt.loadAnns(ann_ids)[0]
                    gt_kps = np.array(gt_ann['keypoints'], dtype=np.float32).reshape(17, 3)

                    # 3) 准备单张图像 Tensor，BGR→RGB，HxWx3→3xHxW，uint8→on cuda
                    img_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
                    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).to(dtype=torch.uint8)

                    # 4) 第一次绘制：绿色真值
                    wb_gt = draw_pose_on_image(img_tensor,
                                               torch.from_numpy(gt_kps),
                                               color=(0, 255, 0))
                    # 从 wandb.Image 获取底层 numpy 图
                    # wandb.Image 对象 .image 属性 即为 numpy 数组（HxWx3）
                    gt_np = wb_gt.image
                    # 转回 RGB Tensor on cuda
                    gt_tensor = torch.from_numpy(gt_np).permute(2, 0, 1).to(dtype=torch.uint8)

                    # 5) 第二次绘制：红色预测
                    wb_pred = draw_pose_on_image(gt_tensor,
                                                 preds[i],
                                                 color=(255, 0, 0))

                    # 6) 最终结果加入列表
                    viz_images.append(wb_pred)

    # COCOeval
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='keypoints')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    mAP, AP50 = float(coco_eval.stats[0]), float(coco_eval.stats[1])

    if was_train:
        model.train()
    return mAP, AP50, viz_images