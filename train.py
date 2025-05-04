import os
import argparse
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import wandb
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from early_stopping_pytorch import EarlyStopping
from utils.coco import prepare_coco_dataset, COCODataset, collate_fn  # 导入数据集类


def setup(rank, world_size):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """清理分布式训练环境"""
    dist.destroy_process_group()


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
    model = OptimizedMobileNetV3().to(rank)
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
            project=f"human_seg_pose{args.project_name}",
            config=vars(args),
            entity=args.entity)

    # 训练循环
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        # 训练阶段
        model.train()
        train_loss = 0.0
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
            if rank == 0 and batch_idx % args.log_interval == 0:
                wandb.log({"train/loss": loss.item()})

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks, hm, paf, _, _, _ in val_loader:
                imgs = imgs.to(rank)
                masks = masks.to(rank)
                hm = hm.to(rank)
                paf = paf.to(rank)

                with autocast(enabled=args.use_fp16):
                    seg_pred, pose_pred = model(imgs)
                    loss = criterion(seg_pred, masks, pose_pred, hm, paf)

                val_loss += loss.item()

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
    parser = argparse.ArgumentParser()
    # 数据参数
    parser.add_argument('--data_dir', default='run/data')
    parser.add_argument('--output_dir', default='run')
    parser.add_argument('--max_samples', type=int, default=None)

    # 训练参数
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--min_delta', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--use_fp16', action='store_true', default=True)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--val_viz_num', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=24)

    # 分布式参数
    parser.add_argument('--dist', action='store_true', default=True)
    parser.add_argument('--local_rank', type=int, default=0)

    # WandB参数
    parser.add_argument('--entity', default='joint_angle')
    parser.add_argument('--project_name', default='_model')

    args = parser.parse_args()

    # 启动多进程训练
    world_size = torch.cuda.device_count()
    mp.spawn(
        main_worker,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )
