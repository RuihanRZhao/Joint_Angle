import os
import time
import random
import argparse
from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pycocotools.cocoeval import COCOeval
import wandb
import cv2
from tqdm import tqdm
from scipy.ndimage import maximum_filter
from PIL import Image

from models.Multi_Pose import MultiPoseNet

from utils.visualization import visualize_coco_keypoints
from utils.loss import PoseLoss
from utils.coco import COCOPoseDataset, COCO_PERSON_SKELETON


NUM_KP=17
# -----------------------
# Evaluation and Visualization
# -----------------------
def evaluate(model, val_loader, device, epoch):
    """
    Evaluate model on COCO val set; return mean AP, AP50, and a list of WandB Images.
    Includes sub-pixel refinement, PAF-based grouping, and custom visualization.
    """
    model.eval()
    coco_gt = val_loader.dataset.coco

    img_ids = []
    vis_list = []

    # 随机选取 n_vis 张图做可视化
    n_vis = getattr(wandb.config, 'n_vis', 3)
    viz_ids = random.sample(val_loader.dataset.img_ids, min(n_vis, len(val_loader.dataset.img_ids)))
    idx_offset = 0

    with torch.no_grad():
        for imgs, _, _ in tqdm(val_loader, desc=f"Epoch: {epoch[0]}/{epoch[1]} Evaluating", unit="batch", leave=False, total=len(val_loader)):
            imgs = imgs.to(device)
            img_metas: List[Dict] = []

            print(0)


            # ready for eval
            for img_id in val_loader.dataset.img_ids:
                th, tw = val_loader.dataset.img_size[1], val_loader.dataset.img_size[0]
                gt_anns = coco_gt.loadAnns(
                    coco_gt.getAnnIds(imgIds=[img_id], catIds=[1], iscrowd=None)
                )

                img_metas.append({
                    'img_id': img_id,
                    'if_viz': img_id in viz_ids,
                    'orig_h': th,
                    'orig_w': tw,
                    'gt_anns': gt_anns,
                })

            heat_pred, paf_pred, results, pred_ann_list = model(imgs, img_metas)

            print(1)

            for img in img_metas:
                # 可视化 GT(green) vs Pred(red)
                if img['if_viz']:

                    img_info = coco_gt.loadImgs([img_id])[0]
                    img_path = os.path.join(
                        val_loader.dataset.root,
                        val_loader.dataset.img_folder,
                        img_info['file_name']
                    )

                    orig_img = cv2.imread(img_path)
                    if orig_img is None:
                        orig_img = np.zeros((img_info['height'], img_info['width'], 3), dtype=np.uint8)

                    h, w = img['orig_h'], img['orig_w']
                    # 先画 GT
                    vis_img = visualize_coco_keypoints(orig_img, img['gt_anns'], COCO_PERSON_SKELETON,(h, w),(0, 255, 0),(0, 255, 0))

                    # 再画 Pred
                    pred_anns = (result for result in pred_ann_list if result.get('img_id') == img['img_id'])

                    print(pred_anns)

                    vis_img = visualize_coco_keypoints(vis_img, pred_anns, COCO_PERSON_SKELETON,(h, w),(0, 0, 255), (0, 0, 255))

                    rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)  # numpy array, H×W×3, uint8
                    vis_list.append(
                        wandb.Image(
                            rgb,
                            caption=f"Image {img['img_id']} – GT(green) vs Pred(red)"
                        )
                    )

            print(2)


    # 6. 运行 COCOeval 并返回指标
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='keypoints')
    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mean_ap = coco_eval.stats[0]   # OKS=0.50:0.95
    ap50    = coco_eval.stats[1]   # OKS=0.50

    return mean_ap, ap50, vis_list

# -----------------------
# Training Loop
# -----------------------
def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    epoch_loss = 0.0

    for batch_idx, (imgs, heatmaps, pafs) in enumerate(tqdm(loader, desc=f"Epoch: {epoch[0]}/{epoch[1]} Training", leave=False)):
        imgs = imgs.to(device)
        heatmaps = heatmaps.to(device)
        pafs = pafs.to(device)

        optimizer.zero_grad()

        # 前向传播
        student_out = model(imgs)

        # 监督损失
        loss = criterion(student_out, (heatmaps, pafs))


        loss.backward()
        # 计算梯度范数
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        optimizer.step()
        epoch_loss += loss.item() * imgs.size(0)

    return epoch_loss / len(loader.dataset)

# -----------------------
# Early Stopp
# -----------------------
class EarlyStopping:
    """
    Early stops training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=5, min_delta=0.0):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored loss to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_ap = None
        self.early_stop = False

    def __call__(self, now_ap):
        if self.best_ap is None:
            self.best_ap = now_ap
        elif now_ap < self.best_ap + self.min_delta:
            # 验证指标没有提升，计数
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # 验证指标有提升，重置计数并更新最佳
            self.best_ap = now_ap
            self.counter = 0


# -----------------------
# Main
# -----------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="轻量化多人姿态估计训练脚本")
    parser.add_argument('--data_root',   type=str,   default='run/data', help='COCO 数据集根目录')
    parser.add_argument('--batch_size',  type=int,   default=256,         help='训练批大小')
    parser.add_argument('--lr',          type=float, default=1e-3,       help='初始学习率')
    parser.add_argument('--epochs',      type=int,   default=1000,       help='训练轮数')
    parser.add_argument('--img_h',       type=int,   default=256,        help='输入图像高度')
    parser.add_argument('--img_w',       type=int,   default=192,        help='输入图像宽度')
    parser.add_argument('--hm_h',        type=int,   default=64,         help='输出热图高度')
    parser.add_argument('--hm_w',        type=int,   default=48,         help='输出热图宽度')
    parser.add_argument('--sigma',       type=int,   default=2,          help='高斯热图 sigma')
    parser.add_argument('--ohkm_k',      type=int,   default=8,          help='OHKM 困难关键点 topK')
    parser.add_argument('--num_workers', type=int,   default=16,         help='DataLoader 线程数')
    parser.add_argument('--patience',    type=int,   default=20,         help='')
    parser.add_argument('--min_delta',   type=float, default=0.000,      help='')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # 创建模型保存目录
    save_dir = 'run/models'
    os.makedirs(save_dir, exist_ok=True)

    # 初始化学生模型、损失、优化器、调度器
    model = MultiPoseNet(num_keypoints=NUM_KP, width_mult=1.0, refine=True).to(device)
    criterion = PoseLoss(ohkm_k=args.ohkm_k)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # 数据集与 DataLoader
    train_ds = COCOPoseDataset(
        root=args.data_root,
        ann_file=os.path.join(args.data_root, 'annotations/person_keypoints_train2017.json'),
        img_folder='train2017',
        img_size=(args.img_h, args.img_w),
        hm_size=(args.hm_h, args.hm_w),
        sigma=args.sigma,
        augment=True
    )
    val_ds = COCOPoseDataset(
        root=args.data_root,
        ann_file=os.path.join(args.data_root, 'annotations/person_keypoints_val2017.json'),
        img_folder='val2017',
        img_size=(args.img_h, args.img_w),
        hm_size=(args.hm_h, args.hm_w),
        sigma=args.sigma,
        augment=False
    )
    print()

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    early_stopper = EarlyStopping(patience=10, min_delta=0.0001)

    print(f"-----------CONFIG----------------")
    for name, value in vars(args).items():
        print(f"  {name:15s} = {value}")

    print(f"  Train samples   = {len(train_ds)}")
    print(f"  Val samples     = {len(val_ds)}")
    print(f"  Device          = {device}")
    print(f"---------------------------------")

    wandb.init(project='Multi_Pose_test', entity="joint_angle",config=vars(args))

    best_ap = 0.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        # 训练
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, (epoch, args.epochs)
        )
        scheduler.step()

        # 验证
        mean_ap, ap50, vis_images = evaluate(model, val_loader, device, (epoch, args.epochs))

        elapsed = time.time() - start
        wandb.log({
            "train/epoch_loss": train_loss,
            "train/lr": scheduler.get_last_lr()[0],
            "val/mAP": mean_ap,
            "val/AP50": ap50,
            "viz/examples": vis_images,
            "epoch/time":elapsed
        }, step=epoch)

        print(f"[Epoch {epoch}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f}  "
              f"mAP: {mean_ap:.4f}  AP50: {ap50:.4f}  "
              f"Time: {elapsed:.1f}s")

        # 保存最佳模型
        if mean_ap > best_ap:
            best_ap = mean_ap
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))

        wandb.log({
            "train/epoch_loss": train_loss,
            "train/lr": scheduler.get_last_lr()[0],
            "val/mAP": mean_ap,
            "val/AP50": ap50,
            "viz/examples": vis_images,
            "epoch/time": elapsed,
        }, step=epoch)

        for name, param in model.named_parameters():
            wandb.log({f"params/{name}": wandb.Histogram(param.detach().cpu().numpy())},
                      step=epoch)

        if early_stopper(mean_ap):
            print(f"Early stopping at epoch {epoch}")
            break