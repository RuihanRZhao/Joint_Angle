import os

import cv2
import random
import numpy as np
import torch
from torch.distributed import destroy_process_group
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import wandb

from utils.visualization import SKELETON
from models.SegKP_Model import SegmentKeypointModel
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from early_stopping_pytorch import EarlyStopping

from utils.coco import prepare_coco_dataset, COCODataset, collate_fn  # 导入数据集类
from utils.loss import criterion
from models.SegKP_Model import PosePostProcessor

from config import arg_test, arg_real



def create_mask_visual(mask):
    """创建分割掩膜可视化"""
    # 使用黑色背景上的红色掩膜
    vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
    vis[mask > 0] = [255, 0, 0]
    return vis


def create_pose_visual(kps_list, image_size, skeleton):
    """创建多人关键点可视化"""
    h, w = image_size
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    for person_kps in kps_list:
        # 绘制骨骼连线
        for (a, b) in skeleton:
            if a >= len(person_kps) or b >= len(person_kps):
                continue
            x1, y1 = person_kps[a]
            x2, y2 = person_kps[b]
            if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)),
                         (0, 255, 0), 2)

        # 绘制关键点
        for (x, y) in person_kps:
            if x > 0 and y > 0:
                cv2.circle(vis, (int(x), int(y)), 3, (0, 0, 255), -1)

    return vis


def log_samples(
    seg_gts, seg_preds,
    pose_gts, pose_preds,
    original_sizes,  # 原始图像尺寸列表 [(H,W), ...]
    epoch,
    skeleton=SKELETON
):
    """记录样本可视化对比结果到WandB"""
    indices = random.sample(range(len(seg_gts)), min(2, len(seg_gts)))

    table = wandb.Table(columns=[
        "GT Segmentation", "Pred Segmentation",
        "GT Keypoints", "Pred Keypoints"
    ])

    post_processor = PosePostProcessor()

    for idx in indices:
        orig_h, orig_w = original_sizes[idx]

        gt_mask = seg_gts[idx].cpu().numpy().squeeze()
        gt_mask = cv2.resize(gt_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        gt_seg = create_mask_visual(gt_mask)

        pred_logit = seg_preds[idx].cpu().numpy().squeeze()
        pred_mask = cv2.resize(pred_logit, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        pred_mask = (torch.sigmoid(torch.from_numpy(pred_mask)) > 0.5).numpy()
        pred_seg = create_mask_visual(pred_mask)

        gt_heatmap = pose_gts[idx].cpu().numpy()
        gt_heatmap = torch.nn.functional.interpolate(
            torch.from_numpy(gt_heatmap).unsqueeze(0),
            size=(orig_h, orig_w),
            mode='bilinear',
            align_corners=False
        )[0].numpy()
        gt_kps = post_processor(gt_heatmap[None])[0]
        gt_pose = create_pose_visual(gt_kps, (orig_h, orig_w), skeleton)

        pred_heatmap = pose_preds[idx].cpu().numpy()
        pred_heatmap = torch.nn.functional.interpolate(
            torch.from_numpy(pred_heatmap).unsqueeze(0),
            size=(orig_h, orig_w),
            mode='bilinear',
            align_corners=False
        )[0].numpy()
        pred_kps = post_processor(pred_heatmap[None])[0]
        pred_pose = create_pose_visual(pred_kps, (orig_h, orig_w), skeleton)

        table.add_data(
            wandb.Image(gt_seg),
            wandb.Image(pred_seg),
            wandb.Image(gt_pose),
            wandb.Image(pred_pose)
        )

    wandb.log({f"Validation Samples Epoch {epoch}": table})


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据集加载
    train_samples = prepare_coco_dataset(
        args.data_dir, split='train', max_samples=args.max_samples)
    val_samples = prepare_coco_dataset(
        args.data_dir, split='val', max_samples=args.max_samples)

    train_dataset = COCODataset(train_samples)
    val_dataset = COCODataset(val_samples)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True, collate_fn=collate_fn)

    # 模型和优化器
    model = SegmentKeypointModel().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 学习率调度器
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=args.epochs * len(train_loader),
        cycle_mult=1.0,
        max_lr=args.lr,
        min_lr=1e-6,
        warmup_steps=args.warmup_epochs * len(train_loader),
        gamma=1.0
    )

    # 混合精度
    scaler = GradScaler(device="cuda", enabled=args.use_fp16)

    # 提前停止
    early_stopping = EarlyStopping(
        patience=args.patience, delta=args.min_delta)

    # WandB
    if args.use_wandb:
        wandb.init(
            project=args.project_name,
            config=vars(args),
            entity=args.entity,
            mode='offline' if args.offline_wandb else 'online'
        )

    # 训练循环
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm.tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{args.epochs} [Train]")

        for imgs, masks, hm, paf, _, _, _ in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            hm = hm.to(device)
            paf = paf.to(device)

            optimizer.zero_grad()
            with autocast(device_type="cuda", enabled=args.use_fp16):
                seg_pred, pose_pred = model(imgs)
                loss = criterion(seg_pred, masks, pose_pred, hm, paf)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if args.use_wandb and (pbar.n % args.log_interval == 0):
                wandb.log({"train/loss": loss.item(), "epoch": epoch})

        pbar.close()
        avg_train_loss = train_loss / len(train_loader)

        # 验证
        model.eval()
        val_loss = 0.0
        sample_seg_gts, sample_seg_preds = [], []
        sample_pose_gts, sample_pose_preds = [], []
        sample_sizes = []

        with torch.no_grad():
            for imgs, masks, hm, paf, _, _, sizes in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                hm = hm.to(device)
                paf = paf.to(device)

                with autocast(device_type="cuda", enabled=args.use_fp16):
                    seg_pred, pose_pred = model(imgs)
                    loss = criterion(seg_pred, masks, pose_pred, hm, paf)

                val_loss += loss.item()

                if len(sample_seg_gts) < args.val_viz_num:
                    sample_seg_gts.extend(masks.cpu().unbind())
                    sample_seg_preds.extend(seg_pred.cpu().unbind())
                    sample_pose_gts.extend(hm.cpu().unbind())
                    sample_pose_preds.extend(pose_pred.cpu().unbind())
                    sample_sizes.extend(sizes)

        avg_val_loss = val_loss / len(val_loader)

        if args.use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/avg_loss": avg_train_loss,
                "val/avg_loss": avg_val_loss
            })
            log_samples(
                seg_gts=sample_seg_gts,
                seg_preds=sample_seg_preds,
                pose_gts=sample_pose_gts,
                pose_preds=sample_pose_preds,
                original_sizes=sample_sizes,
                epoch=epoch
            )

        # EarlyStopping检查
        if early_stopping(avg_val_loss, model):
            print("Early stopping triggered")
            break

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model_final.pth"))


if __name__ == "__main__":
    args = arg_test()
    # args = arg_real()

    # 启动多进程训练
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
    destroy_process_group()
