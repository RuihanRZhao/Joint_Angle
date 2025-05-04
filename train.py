import os
import time
import json
import pprint
import argparse
import logging
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import wandb
from tqdm import tqdm, trange

from pycocotools.coco import COCO

from models.teacher_model import TeacherModel
from models.student_model import StudentModel
from utils.loss import SegmentationLoss, HeatmapLoss, PAFLoss, DistillationLoss
from utils.coco import prepare_coco_dataset, COCODataset, collate_fn
from utils.visualization import overlay_mask, draw_heatmap, draw_paf, draw_keypoints_linked_multi
from utils.metrics import validate_all
from utils.wandbLogger import WandbLogger

# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def parse_args():
    parser = argparse.ArgumentParser(description="Seg→Pose Distill Training with Advanced Features")
    parser.add_argument('--data_dir', default='run/data', help='COCO 数据集根目录')
    parser.add_argument('--output_dir', default='run', help='输出文件目录')
    parser.add_argument('--teacher_epochs', type=int, default=3, help='教师模型训练轮数')
    parser.add_argument('--student_epochs', type=int, default=3, help='学生蒸馏训练轮数')
    parser.add_argument('--batch_size', type=int, default=4, help='训练批大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='初始学习率')
    # scheduler params
    parser.add_argument('--scheduler', choices=['plateau', 'cosine'], default='plateau', help='学习率调度策略')
    parser.add_argument('--plateau_factor', type=float, default=0.5, help='ReduceLROnPlateau 因子')
    parser.add_argument('--plateau_patience', type=int, default=3, help='ReduceLROnPlateau 耐心轮数')
    parser.add_argument('--cosine_tmax', type=int, default=50, help='CosineAnnealingLR T_max')
    parser.add_argument('--cosine_eta_min', type=float, default=1e-6, help='CosineAnnealingLR 最低学习率')
    # EarlyStopping
    parser.add_argument('--min_delta', type=float, default=1e-4, help='EarlyStopping 最小改进阈值')
    parser.add_argument('--patience', type=int, default=7, help='EarlyStopping 耐心值（轮）')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='线性 warmup 的 epoch 数')
    parser.add_argument('--val_viz_num', type=int, default=3, help='每轮上传至 WandB 的验证样本数量')
    parser.add_argument('--num_workers', type=int, default=1, help='DataLoader 并行 worker 数量')
    
    # 混合精度
    parser.add_argument('--use_fp16', action='store_true', default=True, help='启用 torch.cuda.amp 混合精度')
    # debug
    parser.add_argument('--max_samples', type=int, default=10, help='最大加载样本数（快速验证）')
    # distributed
    parser.add_argument('--local_rank', type=int, default=0, help='分布式训练本地 GPU 编号')
    parser.add_argument('--dist', action='store_true', help='是否启用分布式训练')
    # wandb
    parser.add_argument('--entity', default='joint_angle', help='WandB 实体名（用户名或团队）')

    parser.add_argument('--device', default=('cuda' if torch.cuda.is_available() else 'cpu'), help='运行设备')

    parser.add_argument('--project_name', default='', help='Project名字后缀')

    return parser.parse_args()


def parse_args_real():
    parser = argparse.ArgumentParser(description="Seg→Pose Distill Training for B200 (512×512)")

    # 数据与输出
    parser.add_argument('--data_dir', default='run/data', help='COCO 数据集根目录')
    parser.add_argument('--output_dir', default='run', help='输出文件目录')

    # 训练轮数
    parser.add_argument('--teacher_epochs', type=int, default=250, help='教师模型训练轮数')
    parser.add_argument('--student_epochs', type=int, default=100, help='学生蒸馏训练轮数')

    # 大分辨率下的 batch size
    parser.add_argument('--batch_size', type=int, default=64, help='训练批大小 (480x480 建议 8–16)')

    # 学习率
    parser.add_argument('--lr', type=float, default=1e-4, help='初始学习率')

    # 学习率调度器参数
    parser.add_argument('--scheduler', choices=['plateau', 'cosine'], default='plateau',
                        help='学习率调度策略')
    parser.add_argument('--plateau_factor', type=float, default=0.5, help='ReduceLROnPlateau 因子')
    parser.add_argument('--plateau_patience', type=int, default=5, help='ReduceLROnPlateau 耐心轮数')
    parser.add_argument('--cosine_tmax', type=int, default=100, help='CosineAnnealingLR T_max')
    parser.add_argument('--cosine_eta_min', type=float, default=1e-6, help='CosineAnnealingLR 最低学习率')

    # EarlyStopping
    parser.add_argument('--min_delta', type=float, default=1e-5, help='EarlyStopping 最小改进阈值')
    parser.add_argument('--patience', type=int, default=25, help='EarlyStopping 耐心值（轮）')
    
    # 混合精度
    parser.add_argument('--use_fp16', action='store_true', default=True, help='启用 torch.cuda.amp 混合精度')

    # Warmup & 可视化
    parser.add_argument('--warmup_epochs', type=int, default=10, help='线性 warmup 的 epoch 数')
    parser.add_argument('--val_viz_num', type=int, default=2, help='每轮上传至 WandB 的验证样本数量')

    # DataLoader 并行
    parser.add_argument('--num_workers', type=int, default=24, help='DataLoader 并行 worker 数量')

    # 调试用
    parser.add_argument('--max_samples', type=int, default=None, help='最大加载样本数（快速验证）')

    # 分布式训练
    parser.add_argument('--local_rank', type=int, default=0, help='分布式训练本地 GPU 编号')
    parser.add_argument('--dist', action='store_true', help='是否启用分布式训练')

    # WandB
    parser.add_argument('--entity', default='joint_angle', help='WandB 实体名（用户名或团队）')
    parser.add_argument('--device', default=('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='运行设备')

    parser.add_argument('--project_name', default='_model', help='Project名字后缀')

    return parser.parse_args()


def setup_distributed(args):
    if args.dist:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size, rank = 1, 0
    return world_size, rank


def get_models_and_optim(args, device):
    # Teacher
    teacher = TeacherModel().to(device)
    opt_t = torch.optim.Adam(teacher.parameters(), lr=args.lr)
    if args.scheduler == 'plateau':
        sched_t = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_t, mode='min', factor=args.plateau_factor, patience=args.plateau_patience)
    else:
        sched_t = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_t, T_max=args.cosine_tmax, eta_min=args.cosine_eta_min)
    es_t = {'best_loss': float('inf'), 'counter': 0}
    # Student
    student = StudentModel().to(device)
    opt_s = torch.optim.Adam(student.parameters(), lr=args.lr)
    if args.scheduler == 'plateau':
        sched_s = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_s, mode='min', factor=args.plateau_factor, patience=args.plateau_patience)
    else:
        sched_s = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_s, T_max=args.cosine_tmax, eta_min=args.cosine_eta_min)

    es_s = {'best_loss': float('inf'), 'counter': 0}

    scaler = torch.amp.GradScaler(
        'cuda'
    ) if args.use_fp16 else None
    
    return teacher, opt_t, sched_t, es_t, student, opt_s, sched_s, es_s, scaler


def visualize_batch(imgs, masks,
                    t_seg, s_seg,
                    t_hm, t_paf, t_multi,
                    s_hm=None, s_paf=None, s_multi=None,
                    kps_list=None, vis_list=None, n=3):
    """
    可视化多人体 Bottom-Up 结果
    """
    outputs = []
    B = imgs.size(0)
    for i in range(min(n, B)):
        # 原图
        img_np = (imgs[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        entry = {}
        # GT 分割
        entry['GT_Segmentation'] = overlay_mask(img_np, masks[i, 0].cpu().numpy())
        # Teacher 分割
        entry['Teacher_Segmentation'] = overlay_mask(
            img_np, torch.sigmoid(t_seg[i, 0]).cpu().numpy()
        )
        # Student 分割（若有）
        if s_seg is not None:
            entry['Student_Segmentation'] = overlay_mask(
                img_np, torch.sigmoid(s_seg[i, 0]).cpu().numpy()
            )

        # Teacher 热力图 & PAF
        entry['Teacher_Heatmap'] = draw_heatmap(img_np, t_hm[i, 0].cpu().numpy())
        entry['Teacher_PAF'] = draw_paf(img_np, t_paf[i].cpu().numpy())
        # Teacher 多人体关键点
        entry['Teacher_Keypoints'] = draw_keypoints_linked_multi(
            img_np, t_multi[i], np.ones((len(t_multi[i]),), dtype=int)
        )

        # Student 热力图 & PAF（若有）
        if s_hm is not None:
            entry['Student_Heatmap'] = draw_heatmap(img_np, s_hm[i, 0].cpu().numpy())
        if s_paf is not None:
            entry['Student_PAF'] = draw_paf(img_np, s_paf[i].cpu().numpy())
        # Student 多人体关键点（若有）
        if s_multi is not None:
            entry['Student_Keypoints'] = draw_keypoints_linked_multi(
                img_np, s_multi[i], np.ones((len(s_multi[i]),), dtype=int)
            )

        # GT 多人体关键点（若有）
        if kps_list is not None:
            entry['GT_Keypoints'] = draw_keypoints_linked_multi(
                img_np, kps_list[i], vis_list[i]
            )

        outputs.append(entry)
    return outputs


def run_one_epoch(model: nn.Module,
                  loader: DataLoader,
                  losses: dict,
                  optimizer=None,
                  teacher: nn.Module = None,
                  device: str = 'cpu',
                  scaler: torch.amp.GradScaler = None,
                  args=None):
    """
    执行一个 epoch 的训练或验证

    Args:
        model:      学生模型
        loader:     DataLoader
        losses:     dict, 包含 'seg','hm','paf','distill' 四个损失模块
        optimizer:  优化器；若为 None 则为验证模式
        teacher:    若不为 None，则进行蒸馏
        device:     'cpu' or 'cuda'
        scaler:     AMP GradScaler；若为 None 则不开混合精度
        args:       传入命令行参数，用于读取权重系数等
    Returns:
        dict: 平均 loss 值，key 为 'train/seg','train/hm','train/paf','train/distill'
              或 'validate/...'，取决于是否传入 optimizer
    """
    is_train = optimizer is not None
    header   = 'train' if is_train else 'validate'

    model.train() if is_train else model.eval()
    if teacher is not None:
        teacher.eval()

    # 用于累加
    agg = {f'{header}/seg': 0.0,
           f'{header}/hm':  0.0,
           f'{header}/paf': 0.0,
           f'{header}/distill': 0.0}
    total_samples = 0

    for batch in tqdm(loader, desc=header.capitalize()):
        imgs, masks, hm_lbl, paf_lbl, gt_kps, gt_vis, _ = batch
        imgs, masks = imgs.to(device), masks.to(device)
        hm_lbl, paf_lbl = hm_lbl.to(device), paf_lbl.to(device)
        gt_kps = [k.to(device) if isinstance(k, torch.Tensor)
                  else torch.from_numpy(k).float().to(device)
                  for k in gt_kps]
        gt_vis = [v.to(device) if isinstance(v, torch.Tensor)
                  else torch.from_numpy(v).float().to(device)
                  for v in gt_vis]

        if is_train:
            optimizer.zero_grad()

        # Teacher forward (no grad)
        if teacher is not None:
            with torch.no_grad():
                t_seg, t_hm, t_paf, *_ = teacher(imgs)

        # Student forward with/without AMP autocast
        autocast_ctx = torch.amp.autocast(device_type='cuda') if scaler is not None else torch.no_grad()
        # 如果是训练且有 scaler，则开启 autocast，否则用普通模式
        if is_train and scaler is not None:
            autocast_ctx = torch.amp.autocast(device_type='cuda')
        else:
            autocast_ctx = torch.no_grad() if not is_train else (lambda: (_ for _ in ()).throw)

        with autocast_ctx:
            s_seg, s_hm, s_paf, s_multi = model(imgs)

            # compute loss
        if teacher is None:
            loss_seg = losses['seg'](s_seg, masks)
            loss_hm = losses['hm'](s_hm, hm_lbl)
            loss_paf = losses['paf'](s_paf, paf_lbl)
            loss = loss_seg + loss_hm + loss_paf
            loss_dist = torch.tensor(0.0, device=device)
        else:
            loss, met = losses['distill'](
                (s_seg, s_hm, s_paf),
                (t_seg.detach(), t_hm.detach(), t_paf.detach()),
                (masks, gt_kps, gt_vis)
            )
            loss_seg = met['seg_loss']
            loss_hm = met['hm_loss']
            loss_paf = met['paf_loss']
            loss_dist = met['distill_loss']

        # 反向传播
        if is_train:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        # 累加各项 loss
        bs = imgs.size(0)
        agg[f'{header}/seg']     += met.get('seg_loss',     0.0).item() * bs
        agg[f'{header}/hm']      += met.get('hm_loss',      0.0).item() * bs
        agg[f'{header}/paf']     += met.get('paf_loss',     0.0).item() * bs
        agg[f'{header}/distill'] += met.get('distill_loss', 0.0).item() * bs
        total_samples += bs

    # 取平均
    for k in agg:
        agg[k] /= total_samples

    return agg




if __name__ == '__main__':
    args = parse_args()
    print("======== Training Configuration ========")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("========================================")

    world_size, rank = setup_distributed(args)
    device = torch.device('cuda', args.local_rank) if args.dist else torch.device(args.device)

    # Create output dirs
    os.makedirs(args.output_dir, exist_ok=True)
    for sub in ['vis', 'models']:
        os.makedirs(os.path.join(args.output_dir, sub), exist_ok=True)

    # Data
    sampler = DistributedSampler if args.dist else lambda ds: None

    print(f"Preparing train dataset...")
    train_s = prepare_coco_dataset(args.data_dir, 'train', max_samples=args.max_samples)
    print(f"Preparing val dataset...")
    val_s = prepare_coco_dataset(args.data_dir, 'val', max_samples=args.max_samples)

    train_ds = COCODataset(train_s)
    val_ds = COCODataset(val_s)
    train_ld = DataLoader(train_ds,
                          batch_size=args.batch_size,
                          sampler=sampler(train_ds),
                          shuffle=not args.dist,
                          collate_fn=collate_fn,
                          num_workers=args.num_workers,
                          pin_memory=True)
    val_ld = DataLoader(val_ds,
                        batch_size=1,
                        sampler=sampler(val_ds),
                        shuffle=False,
                        collate_fn=collate_fn,
                        num_workers=args.num_workers,
                        pin_memory=True)

    # Models & optimizers
    teacher, opt_t, sched_t, es_t, student, opt_s, sched_s, es_s, scaler= get_models_and_optim(args, device)

    if args.dist:
        teacher = DDP(teacher, device_ids=[args.local_rank])
        student = DDP(student, device_ids=[args.local_rank])

    seg_fn = SegmentationLoss()  # 例如 BCEWithLogits 内部封装
    hm_fn = HeatmapLoss()  # 负责 keypoint heatmap 的 L2/L1
    paf_fn = PAFLoss()  # 负责 PAF 的 L2/L1
    dist_fn = DistillationLoss()  # 负责蒸馏三项的综合

    # WandB Logger for Teacher
    logger_t = WandbLogger(f"Teacher{args.project_name}", args.entity, config=vars(args))
    try:
        best_pck = 0.0
        best_t = teacher.state_dict()
        for epoch in trange(1, args.teacher_epochs + 1, desc='Teacher'):
            # Warmup LR
            if epoch <= args.warmup_epochs:
                lr = args.lr * epoch / args.warmup_epochs
                for g in opt_t.param_groups:
                    g['lr'] = lr

            # Train & validate
            metrics = run_one_epoch(
                teacher, train_ld,
                {'seg': seg_fn, 'hm': hm_fn, 'paf': paf_fn},
                optimizer=opt_t, teacher=None, device=device, scaler=scaler
            )
            pck_stats = validate_all(
                student, val_ld, device,
                coco_gt_json     = os.path.join(args.data_dir, 'annotations', 'person_keypoints_val2017.json'),
                tmp_results_json = os.path.join(args.output_dir, f'val_results_student_{epoch}.json'),
                thresh_ratio     = 0.05
            )

            logger_t.log({**metrics, **pck_stats, 'epoch': epoch}, step=epoch)

            # Visualization GT vs Teacher
            imgs, masks, hm_lbl, paf_lbl, kps, vis, _ = next(iter(val_ld))
            with torch.no_grad():
                t_seg_out, t_hm, t_paf, t_multi = teacher(imgs.to(device))
            vizs = visualize_batch(
                imgs, masks,
                t_seg_out, None,
                t_hm, t_paf, t_multi,
                s_hm=None, s_paf=None, s_multi=None,
                kps_list=kps, vis_list=vis,
                n=args.val_viz_num
            )
            for i, d in enumerate(vizs):
                logger_t.log({
                    f'viz/teacher_{i}/GT_Segmentation': wandb.Image(d['GT_Segmentation']),
                    f'viz/teacher_{i}/Teacher_Segmentation': wandb.Image(d['Teacher_Segmentation']),
                    f'viz/teacher_{i}/GT_Keypoints': wandb.Image(d['GT_Keypoints']),
                    f'viz/teacher_{i}/Teacher_Keypoints': wandb.Image(d['Teacher_Keypoints'])
                }, step=epoch)

            # Save best teacher
            val_pck = pck_stats['coco/AP50']
            if val_pck > best_pck:
                    best_pck = val_pck
                    best_t = teacher.state_dict()

            torch.cuda.empty_cache()

        torch.save(best_t, os.path.join(args.output_dir, 'models', 'teacher.pth'))
        logger_t.finish()

        # ===== Student Distillation =====
        teacher.load_state_dict(best_t)
        teacher.eval()
        logger_s = WandbLogger(f"Student{args.project_name}", args.entity, config=vars(args))
        best_pck = 0.0
        best_s = student.state_dict()

        for epoch in trange(1, args.student_epochs + 1, desc='Student'):
            if epoch <= args.warmup_epochs:
                lr = args.lr * epoch / args.warmup_epochs
                for g in opt_s.param_groups:
                    g['lr'] = lr

            metrics = run_one_epoch(
                student, train_ld,
                {'distill': dist_fn},
                optimizer=opt_s, teacher=teacher, device=device, scaler=scaler
            )
            pck_stats = validate_all(
                teacher, val_ld, device,
                coco_gt_json   = os.path.join(args.data_dir, 'annotations', 'person_keypoints_val2017.json'),
                tmp_results_json = os.path.join(args.output_dir, 'val_results.json'),
                thresh_ratio  = 0.05
            )
            logger_s.log({**metrics, **pck_stats, 'epoch': epoch}, step=epoch)

            # Visualization GT vs Teacher vs Student
            imgs, masks, hm_lbl, paf_lbl, kps, vis, _ = next(iter(val_ld))
            with torch.no_grad():
                t_seg_out, t_hm, t_paf, t_multi = teacher(imgs.to(device))
                s_seg_out, s_hm, s_paf, s_multi = student(imgs.to(device))
            vizs = visualize_batch(
                imgs, masks,
                t_seg_out, s_seg_out,
                t_hm, t_paf, t_multi,
                s_hm=s_hm, s_paf=s_paf, s_multi=s_multi,
                kps_list=kps, vis_list=vis,
                n=args.val_viz_num
            )
            for i, d in enumerate(vizs):
                logger_s.log({
                    f'viz/student_{i}/GT_Segmentation': wandb.Image(d['GT_Segmentation']),
                    f'viz/student_{i}/Teacher_Segmentation': wandb.Image(d['Teacher_Segmentation']),
                    f'viz/student_{i}/Teacher_Keypoints': wandb.Image(d['Teacher_Keypoints']),
                    f'viz/student_{i}/Student_Segmentation': wandb.Image(d['Student_Segmentation']),
                    f'viz/student_{i}/Student_Keypoints': wandb.Image(d['Student_Keypoints']),
                    f'viz/student_{i}/GT_Keypoints': wandb.Image(d['GT_Keypoints'])
                }, step=epoch)


            val_pck = pck_stats['coco/AP50']
            if val_pck > best_pck:
                    best_pck = val_pck
                    best_t = student.state_dict()

            torch.cuda.empty_cache()

        torch.save(best_s, os.path.join(args.output_dir, 'models', 'student.pth'))
        logger_s.finish()

    except Exception:
        logging.exception("Training failed")
        raise

    finally:
        if args.dist:
            dist.destroy_process_group()
