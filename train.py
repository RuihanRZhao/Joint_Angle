import os

import cv2
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import tqdm
import wandb

from utils.visualization import SKELETON
from models.SegKP_Model import SegmentKeypointModel
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from early_stopping_pytorch import EarlyStopping

from utils.coco import prepare_coco_dataset, COCODataset, collate_fn  # 导入数据集类
from utils.loss import criterion
from utils.visualization import overlay_mask, draw_keypoints_linked_multi
from models.SegKP_Model import PosePostProcessor

from config import arg_test, arg_real


def setup(rank, world_size):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """清理分布式训练环境"""
    dist.destroy_process_group()


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
    # 随机选择2个样本
    indices = random.sample(range(len(seg_gts)), min(2, len(seg_gts)))

    table = wandb.Table(columns=[
        "GT Segmentation", "Pred Segmentation",
        "GT Keypoints", "Pred Keypoints"
    ])

    # 初始化后处理器
    post_processor = PosePostProcessor()

    for idx in indices:
        # 获取原始尺寸
        orig_h, orig_w = original_sizes[idx]

        # 真实分割处理
        gt_mask = seg_gts[idx].cpu().numpy().squeeze()
        gt_mask = cv2.resize(gt_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        gt_seg = create_mask_visual(gt_mask)

        # 预测分割处理
        pred_logit = seg_preds[idx].cpu().numpy().squeeze()
        pred_mask = cv2.resize(pred_logit, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        pred_mask = (torch.sigmoid(torch.from_numpy(pred_mask)) > 0.5).numpy()
        pred_seg = create_mask_visual(pred_mask)

        # 真实关键点处理
        gt_heatmap = pose_gts[idx].cpu().numpy()
        gt_heatmap = F.interpolate(
            torch.from_numpy(gt_heatmap).unsqueeze(0),
            size=(orig_h, orig_w),
            mode='bilinear',
            align_corners=False
        )[0].numpy()
        gt_kps = post_processor(gt_heatmap[None])[0]  # [P,K,2]
        gt_pose = create_pose_visual(gt_kps, (orig_h, orig_w), skeleton)

        # 预测关键点处理
        pred_heatmap = pose_preds[idx].cpu().numpy()
        pred_heatmap = F.interpolate(
            torch.from_numpy(pred_heatmap).unsqueeze(0),
            size=(orig_h, orig_w),
            mode='bilinear',
            align_corners=False
        )[0].numpy()
        pred_kps = post_processor(pred_heatmap[None])[0]
        pred_pose = create_pose_visual(pred_kps, (orig_h, orig_w), skeleton)

        # 添加到表格
        table.add_data(
            wandb.Image(gt_seg),
            wandb.Image(pred_seg),
            wandb.Image(gt_pose),
            wandb.Image(pred_pose)
        )

    wandb.log({f"Validation Samples Epoch {epoch}": table})


def main_worker(rank, world_size, args):
    """每个GPU的工作进程"""
    # 初始化分布式训练
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    # 数据集加载
    train_samples = prepare_coco_dataset(
        args.data_dir, split='train', max_samples=args.max_samples)
    val_samples = prepare_coco_dataset(
        args.data_dir, split='val', max_samples=args.max_samples)

    # 数据加载器
    train_dataset = COCODataset(train_samples)
    val_dataset = COCODataset(val_samples)

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    # 模型初始化
    model = SegmentKeypointModel().to(rank)
    model = DDP(model, device_ids=[rank])

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 学习率调度器
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=args.epochs,
        cycle_mult=1.0,
        max_lr=args.lr,
        min_lr=1e-6,
        warmup_steps=args.warmup_epochs * len(train_loader),
        gamma=1.0
    )

    # 混合精度
    scaler = GradScaler(enabled=args.use_fp16)

    # EarlyStopping
    early_stopping = EarlyStopping(
        patience=args.patience, delta=args.min_delta)

    # 仅在主进程初始化WandB
    if rank == 0:
        wandb.init(
            project=f"{args.project_name}",
            config=vars(args),
            entity=args.entity)

    # 训练循环
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        # 训练阶段
        model.train()
        train_loss = 0.0

        train_pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{args.epochs} [Train]", position=0, leave=True) if rank == 0 else None

        for batch_idx, (imgs, masks, hm, paf, _, _, _) in enumerate(train_loader):
            imgs = imgs.to(rank)
            masks = masks.to(rank)
            hm = hm.to(rank)
            paf = paf.to(rank)

            optimizer.zero_grad()

            with autocast(enabled=args.use_fp16):
                seg_pred, pose_pred = model(imgs)
                loss = criterion(seg_pred, masks, pose_pred, hm, paf)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()

            if rank == 0:
                train_pbar.update(1)
                train_pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}"
                })

                if batch_idx % args.log_interval == 0:
                    wandb.log({"train/loss": loss.item()})

        if rank == 0: train_pbar.close()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        sample_seg_gts = []
        sample_seg_preds = []
        sample_pose_gts = []
        sample_pose_preds = []
        sample_sizes = []

        val_pbar = tqdm(total=len(val_loader), desc=f"Epoch {epoch + 1}/{args.epochs} [Val]", position=0, leave=False) if rank == 0 else None

        with torch.no_grad():
            for batch in val_loader:
                mgs, masks, hm, paf, _, _, sizes = batch

                imgs = imgs.to(rank)
                masks = masks.to(rank)
                hm = hm.to(rank)
                paf = paf.to(rank)

                with autocast(enabled=args.use_fp16):
                    seg_pred, pose_pred = model(imgs)
                    loss = criterion(seg_pred, masks, pose_pred, hm, paf)

                val_loss += loss.item()

                if rank == 0:
                    val_pbar.update(1)

                    if len(sample_seg_gts) < args.val_viz_num:
                        sample_seg_gts.extend(masks.cpu().unbind())
                        sample_seg_preds.extend(seg_pred.cpu().unbind())
                        sample_pose_gts.extend(hm.cpu().unbind())
                        sample_pose_preds.extend(pose_pred.cpu().unbind())
                        sample_sizes.extend(sizes)

        if rank == 0:
            log_samples(
                seg_gts=sample_seg_gts,
                seg_preds=sample_seg_preds,
                pose_gts=sample_pose_gts,
                pose_preds=sample_pose_preds,
                original_sizes=sample_sizes,
                epoch=epoch + 1
            )
            val_pbar.close()


        # 收集所有进程的损失
        torch.distributed.reduce(train_loss, dst=0)
        torch.distributed.reduce(val_loss, dst=0)

        if rank == 0:
            avg_train_loss = train_loss / (len(train_loader) * world_size)
            avg_val_loss = val_loss / (len(val_loader) * world_size)

            wandb.log({
                "epoch": epoch + 1,
                "train/avg_loss": avg_train_loss,
                "val/avg_loss": avg_val_loss
            })

            # EarlyStopping检查
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

    # 清理分布式环境
    cleanup()


if __name__ == "__main__":
    args = arg_test()
    # args = arg_real()

    # 启动多进程训练
    world_size = torch.cuda.device_count()
    mp.spawn(
        main_worker,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )
