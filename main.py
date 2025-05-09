import os
import time
import random
import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image
import torchvision.transforms as T
import wandb
import cv2
from tqdm import tqdm
from scipy.ndimage import maximum_filter

from utils.loss import PoseLoss
from utils.evaluate import evaluate
from utils.coco import ensure_coco_data, COCOPoseDataset
from utils.train_utils import train_one_epoch, ModelEMA

# -----------------------
# 全局定义（COCO骨架拓扑等）
# -----------------------
from utils.coco import COCO_PERSON_SKELETON
NUM_LIMBS = len(COCO_PERSON_SKELETON)

# -----------------------
# 模型定义
# -----------------------
# 提示: MultiPoseNet 定义在 models/Multi_Pose.py (需包含精细化输出)
try:
    # 如果存在 models/Multi_Pose.py 模块，则从中导入 MultiPoseNet 类
    from models.Multi_Pose import MultiPoseNet
except ImportError:
    # 如果未提供 models/Multi_Pose.py，则需要手动实现 MultiPoseNet 类
    # 这里假设 MultiPoseNet 已在相应模块中实现，具有 refine=True 输出精细化结果
    raise ImportError("请确保在 models/Multi_Pose.py 中定义 MultiPoseNet 类")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多人姿态估计训练脚本 (带精细化模型和改进训练策略)")
    parser.add_argument('--data_root',          type=str,               default='run/data', help='COCO数据集根目录路径')
    parser.add_argument('--batch_size',         type=int,               default=32,         help='训练批次大小')
    parser.add_argument('--lr',                 type=float,             default=1e-3,       help='初始学习率(作为OneCycleLR的最大值)')
    parser.add_argument('--epochs',             type=int,               default=50,         help='训练轮数')
    parser.add_argument('--img_h',              type=int,               default=256,        help='输入图像高度')
    parser.add_argument('--img_w',              type=int,               default=256,        help='输入图像宽度')
    parser.add_argument('--hm_h',               type=int,               default=64,         help='输出热图高度')
    parser.add_argument('--hm_w',               type=int,               default=64,         help='输出热图宽度')
    parser.add_argument('--sigma',              type=int,               default=2,          help='高斯热图Sigma值')
    parser.add_argument('--ohkm_k',             type=int,               default=8,          help='OHKM中选取的困难关键点数')
    parser.add_argument('--n_vis',              type=int,               default=3,          help='每轮epoch可视化的验证示例数')
    parser.add_argument('--num_workers_train',  type=int,               default=8,          help='训练DataLoader的worker数量')
    parser.add_argument('--num_workers_val',    type=int,               default=4,          help='验证DataLoader的worker数量')
    parser.add_argument('--grad_clip',          type=float,             default=5.0,        help='梯度裁剪的最大L2范数')
    parser.add_argument('--seed',               type=int,               default=42,         help='随机种子（用于训练可重复性）')
    parser.add_argument('--use_amp',            action='store_true',    default=False,      help="启用 AMP 混合精度训练")  # 改为正向参数 use_amp，默认False表示不使用AMP
    parser.add_argument('--use_ema',            action='store_true',    default=False,      help="启用 EMA 模型权重平滑")  # 改为正向参数 use_ema，默认False表示不使用EMA
    parser.add_argument('--grad_clip',          type=float,             default=5.0,        help='梯度裁剪阈值')

    args = parser.parse_args()

    # 设置随机种子确保可复现
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # 创建模型保存目录
    os.makedirs('run/models', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化WandB日志
    wandb.init(project='final', entity='joint_angle', config=vars(args))
    config = wandb.config

    # 模型、损失函数、优化器、学习率调度器
    model = MultiPoseNet(num_keypoints=17, refine=True).to(device)  # 使用精细化多人姿态模型
    criterion = PoseLoss(ohkm_k=config.ohkm_k)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    # 计算OneCycleLR需要的总steps数
    steps_per_epoch = None
    try:
        # 尽量准确获取训练steps
        dummy_dataset = COCOPoseDataset(root=config.data_root, ann_file=os.path.join(config.data_root, 'annotations/person_keypoints_train2017.json'),
                                        img_folder='train2017', img_size=(config.img_h, config.img_w), hm_size=(config.hm_h, config.hm_w), sigma=config.sigma)
        steps_per_epoch = int(np.ceil(len(dummy_dataset) / config.batch_size))
    except Exception as e:
        # 如数据集无法提前加载，则通过DataLoader确定
        steps_per_epoch = None
    if steps_per_epoch is None:
        # 若无法提前计算，则初始化一个DataLoader获取长度
        temp_ds = COCOPoseDataset(root=config.data_root, ann_file=os.path.join(config.data_root, 'annotations/person_keypoints_train2017.json'),
                                  img_folder='train2017', img_size=(config.img_h, config.img_w), hm_size=(config.hm_h, config.hm_w), sigma=config.sigma)
        steps_per_epoch = int(np.ceil(len(temp_ds) / config.batch_size))
    total_steps = config.epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.lr, total_steps=total_steps, epochs=config.epochs, steps_per_epoch=steps_per_epoch, pct_start=0.3, anneal_strategy='cos')

    # 自动混合精度和EMA设置

    use_amp = args.use_amp  # 更改：直接使用 args.use_amp（True表示使用AMP）。
    ema_model = ModelEMA(model).to(device) if args.use_ema else None  # 更改：根据 args.use_ema 决定是否初始化EMA（原本使用args.no_ema反逻辑）
    amp_enabled = args.use_amp and torch.cuda.is_available()
    scaler = GradScaler(enabled=amp_enabled)
    ema_decay = 0.999  # EMA权重衰减率

    # 准备数据集和数据加载器
    train_ds = COCOPoseDataset(root=config.data_root,
                               ann_file=os.path.join(config.data_root, 'annotations/person_keypoints_train2017.json'),
                               img_folder='train2017',
                               img_size=(config.img_h, config.img_w),
                               hm_size=(config.hm_h, config.hm_w),
                               sigma=config.sigma)
    val_ds = COCOPoseDataset(root=config.data_root,
                             ann_file=os.path.join(config.data_root, 'annotations/person_keypoints_val2017.json'),
                             img_folder='val2017',
                             img_size=(config.img_h, config.img_w),
                             hm_size=(config.hm_h, config.hm_w),
                             sigma=config.sigma)
    print(f"Train Samples: {len(train_ds)}")
    print(f"Val Samples: {len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers_train, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers_val, pin_memory=True)

    # 记录模型参数总量
    total_params = sum(p.numel() for p in model.parameters())
    wandb.log({"num_parameters": total_params}, step=0)

    best_ap = 0.0
    best_ap50 = 0.0

    ema_updater = ModelEMA(model, decay=0.999) if args.use_ema else None

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # 单个 epoch 训练（AMP、梯度裁剪、EMA 已封装在函数中）
        avg_train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            use_amp=args.use_amp,
            grad_clip=args.grad_clip,
            ema=ema_updater
        )

        # 验证（使用 EMA 模型推理更稳定）
        val_model = ema_updater.ema_model if ema_updater else model
        val_mAP, val_AP50, val_vis_images, avg_val_loss = evaluate(val_model, val_loader, device, criterion)

        epoch_time = time.time() - epoch_start

        # 记录epoch级别的指标
        metrics = {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss if avg_val_loss is not None else 0.0,
            'val_mAP': val_mAP,
            'val_AP50': val_AP50,
            'epoch_time': epoch_time,
            'lr': scheduler.get_last_lr()[0],
            'gpu_mem_used': torch.cuda.memory_allocated(device) / (1024**3) if torch.cuda.is_available() else 0
        }
        wandb.log(metrics, commit=False)
        # 记录模型参数和梯度分布直方图
        for name, param in model.named_parameters():
            if param.requires_grad:
                wandb.log({f"param/{name}": wandb.Histogram(param.detach().cpu().numpy())}, commit=False)
                if param.grad is not None:
                    wandb.log({f"grad/{name}": wandb.Histogram(param.grad.detach().cpu().numpy())}, commit=False)
        # 提交所有日志
        wandb.log({}, commit=True)
        # 保存当前epoch的模型checkpoint并上传为WandB Artifact
        ckpt_path = f'run/models/epoch{epoch}.pth'
        torch.save(model.state_dict(), ckpt_path)
        artifact = wandb.Artifact(f'pose-model-epoch{epoch}', type='model')
        artifact.add_file(ckpt_path)
        wandb.log_artifact(artifact)
        # 上传验证可视化示例图片
        if len() > 0:
            wandb.log({'val_examples': val_vis_images})
        # 保存并更新最佳模型
        if val_mAP > best_ap:
            best_ap = val_mAP
            best_ap50 = val_AP50
            best_path = 'run/models/best_model.pth'
            # 保存EMA模型权重为最佳模型 (如使用EMA)，否则保存当前模型权重
            if use_ema:
                torch.save(ema_model.state_dict(), best_path)
            else:
                torch.save(model.state_dict(), best_path)
            wandb.run.summary["best_mAP"] = best_ap
            wandb.run.summary["best_AP50"] = best_ap50
            print(f"Epoch {epoch}: New best mAP {best_ap:.4f}, AP50 {best_ap50:.4f}, model saved.")
        else:
            print(f"Epoch {epoch}: mAP {val_mAP:.4f}, AP50 {val_AP50:.4f} (best mAP {best_ap:.4f})")

    # 训练结束，记录最佳结果到wandb.summary并结束wandb
    wandb.summary["best_mAP"] = best_ap
    wandb.summary["best_AP50"] = best_ap50
    wandb.finish()