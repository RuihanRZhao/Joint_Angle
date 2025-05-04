import os
import time
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
from utils.Evaluator import SegmentationEvaluator,PoseEvaluator
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

    wandb.log({f"Validation Samples Epoch {epoch}": table}, step=epoch)

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据加载
    train_samples = prepare_coco_dataset(args.data_dir, split='train', max_samples=args.max_samples)
    val_samples   = prepare_coco_dataset(args.data_dir, split='val',   max_samples=args.max_samples)
    train_loader  = DataLoader(COCODataset(train_samples), batch_size=args.batch_size,
                               shuffle=True, num_workers=args.num_workers,
                               pin_memory=True, collate_fn=collate_fn)
    val_loader    = DataLoader(COCODataset(val_samples),   batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers,
                               pin_memory=True, collate_fn=collate_fn)

    # 模型、优化器、调度
    model     = SegmentKeypointModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=args.epochs * len(train_loader),
        cycle_mult=1.0,
        max_lr=args.lr,
        min_lr=1e-6,
        warmup_steps=args.warmup_epochs * len(train_loader),
        gamma=1.0
    )

    # 混合精度、早停
    scaler = GradScaler(enabled=args.use_fp16)
    early_stopping = EarlyStopping(patience=args.patience, delta=args.min_delta)

    # WandB
    wandb.init(project=args.project_name, config=vars(args), entity=args.entity)
    post_processor = PosePostProcessor()

    for epoch in range(1, args.epochs + 1):
        # 记录时间
        epoch_start = time.time()

        # 训练
        model.train()
        train_loss = 0.0
        pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{args.epochs} [Train]")

        for batch_idx, (imgs, masks, hm, paf, _, _, _) in enumerate(train_loader, start=1):
            batch_start = time.time()
            imgs, masks, hm, paf = imgs.to(device), masks.to(device), hm.to(device), paf.to(device)

            optimizer.zero_grad()
            if args.use_fp16:
                with autocast(device_type='cuda', enabled=True):
                    seg_pred, pose_pred = model(imgs)
                    loss = criterion(seg_pred, masks, pose_pred, hm, paf)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                seg_pred, pose_pred = model(imgs)
                loss = criterion(seg_pred, masks, pose_pred, hm, paf)
                loss.backward()
                optimizer.step()

            scheduler.step()
            train_loss += loss.item()

            # 计算指标
            batch_time = time.time() - batch_start
            current_lr = optimizer.param_groups[0]['lr']
            total_norm = sum(p.grad.data.norm(2).item()**2 for p in model.parameters() if p.grad is not None)**0.5

            if args.use_wandb:
                wandb.log({
                    'train/batch_loss': loss.item(),
                    'train/lr': current_lr,
                    'train/grad_norm': total_norm,
                    'train/batch_time': batch_time
                }, step=(epoch-1)*len(train_loader) + batch_idx)

            pbar.update(1)
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{current_lr:.2e}", 'bt': f"{batch_time:.2f}s"})

        pbar.close()
        avg_train_loss = train_loss / len(train_loader)
        epoch_time = time.time() - epoch_start

        # 验证
        val_start = time.time()
        model.eval()
        val_loss = 0.0

        all_pred_masks, all_gt_masks = [], []
        all_pred_kps, all_gt_kps   = [], []
        sample_seg_gts, sample_seg_preds = [], []
        sample_pose_gts, sample_pose_preds = [], []
        sample_sizes = []

        with torch.no_grad():
            for imgs, masks, hm, paf, _, _, sizes in val_loader:
                imgs, masks, hm, paf = imgs.to(device), masks.to(device), hm.to(device), paf.to(device)
                with autocast(device_type='cuda', enabled=args.use_fp16):
                    seg_pred, pose_pred = model(imgs)
                    loss = criterion(seg_pred, masks, pose_pred, hm, paf)
                val_loss += loss.item()

                # 收集用于 evaluator 和可视化
                pred_mask = (torch.sigmoid(seg_pred) > 0.5).cpu().numpy()
                for i, pm in enumerate(pred_mask):
                    h, w = sizes[i]
                    pm_resized = cv2.resize(pm.squeeze(), (w, h), interpolation=cv2.INTER_NEAREST)
                    all_pred_masks.append(pm_resized)
                    all_gt_masks.append(masks[i].cpu().numpy().squeeze())

                pred_kps = post_processor(pose_pred.cpu().numpy(), sizes)
                gt_kps   = post_processor(hm.cpu().numpy(), sizes)
                all_pred_kps.extend(pred_kps)
                all_gt_kps.extend(gt_kps)

                if len(sample_seg_gts) < args.val_viz_num:
                    sample_seg_gts.extend(masks.cpu().unbind())
                    sample_seg_preds.extend(seg_pred.cpu().unbind())
                    sample_pose_gts.extend(hm.cpu().unbind())
                    sample_pose_preds.extend(pose_pred.cpu().unbind())
                    sample_sizes.extend(sizes)

        avg_val_loss = val_loss / len(val_loader)
        val_time = time.time() - val_start

        # 计算 evaluator 指标
        seg_iou  = SegmentationEvaluator(all_pred_masks, all_gt_masks)
        kp_acc   = PoseEvaluator(all_pred_kps, all_gt_kps)

        # WandB 日志
        if args.use_wandb:
            wandb.log({
                'train/avg_loss': avg_train_loss,
                'val/avg_loss': avg_val_loss,
                'train/epoch_time': epoch_time,
                'val/epoch_time': val_time,
                'val/seg_iou': seg_iou,
                'val/kp_accuracy': kp_acc,
                **{f"model/{n.replace('.', '/')}_hist": wandb.Histogram(p.detach().cpu().numpy())
                   for n, p in model.named_parameters()}
            }, step=epoch)

            wandb.log({f"Validation Samples Epoch {epoch}":
                       log_samples(
                           seg_gts=sample_seg_gts,
                           seg_preds=sample_seg_preds,
                           pose_gts=sample_pose_gts,
                           pose_preds=sample_pose_preds,
                           original_sizes=sample_sizes,
                           epoch=epoch
                       )}, step=epoch)

        # EarlyStopping
        if early_stopping(avg_val_loss, model):
            print("Early stopping triggered")
            break

    # 保存模型
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model_final.pth"))

    destroy_process_group()


if __name__ == "__main__":
    args = arg_test()
    # args = arg_real()

    # 启动多进程训练
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
    destroy_process_group()
