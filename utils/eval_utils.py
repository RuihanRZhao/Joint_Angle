import os
import random
import torch
import wandb
import cv2
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
    coco_gt = COCO(ann_file)

    results = []
    viz_images = []
    # 预选随机可视化索引
    total_samples = len(val_loader.dataset)
    viz_idxs = set(random.sample(range(total_samples), n_viz))

    device = next(model.parameters()).device

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(val_loader), desc="[Evaluating]"):
            images, meta = batch
            images = images.to(device)

            # 模型前向得出热图，形状 (B,17,H,W)
            heatmaps = model(images)
            if isinstance(heatmaps, (tuple, list)):
                # 如果模型输出初始和精修两级 heatmap，取第二级
                heatmaps = heatmaps[-1]

            # 批量解码关键点，得到 (B,17,3)
            kpts_batch = batch_heatmaps_to_keypoints(heatmaps)
            kpts_batch = kpts_batch.cpu()

            # 遍历批次样本
            B = kpts_batch.shape[0]
            for i in range(B):
                image_id = int(meta['image_id'][i])
                kpts = kpts_batch[i]  # (17,3)
                # 构建 COCO 格式结果字典
                kpts_list = kpts.view(-1).tolist()  # flatten to length 51
                score = float(kpts[:,2].mean())
                results.append({
                    "image_id": image_id,
                    "category_id": 1,
                    "keypoints": kpts_list,
                    "score": score
                })
                # 可视化随机挑选的样本
                global_idx = batch_idx * val_loader.batch_size + i
                if global_idx in viz_idxs:
                    # 读取原始图像
                    info = coco_gt.loadImgs(image_id)[0]
                    img_path = os.path.join(val_image_dir, info['file_name'])
                    orig = cv2.imread(img_path)
                    drawn = draw_pose_on_image(orig, kpts[:, :2].numpy(), kpts[:, 2].numpy())
                    viz_images.append(wandb.Image(drawn, caption=f"ID:{image_id}"))

    # 用 COCOeval 评估 AP
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='keypoints')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    mAP, AP50 = float(coco_eval.stats[0]), float(coco_eval.stats[1])

    # 恢复模型状态
    if was_train:
        model.train()

    return mAP, AP50, viz_images