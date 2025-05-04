import os
import time
import random
import argparse
import logging
import numpy as np
import cv2
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import wandb
from tqdm import tqdm, trange

from models.teacher_model import TeacherModel
from models.student_model import StudentModel
from utils.loss import SegmentationLoss, KeypointsLoss, DistillationLoss
from utils.coco import prepare_coco_dataset, COCODataset, collate_fn
from utils.visualization import draw_keypoints_linked
from utils.wandbLogger import WandbLogger
from utils.metrics import compute_miou, compute_pck


# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Seg→Pose Distill Training with Advanced Features")
    parser.add_argument('--data_dir', default='run/data', help='COCO 数据集根目录')
    parser.add_argument('--output_dir', default='run', help='输出文件目录')
    parser.add_argument('--teacher_epochs', type=int, default=3, help='教师模型训练轮数')
    parser.add_argument('--student_epochs', type=int, default=3, help='学生蒸馏训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='训练批大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='初始学习率')
    # scheduler params
    parser.add_argument('--scheduler', choices=['plateau','cosine'], default='plateau', help='学习率调度策略')
    parser.add_argument('--plateau_factor', type=float, default=0.5, help='ReduceLROnPlateau 因子')
    parser.add_argument('--plateau_patience', type=int, default=3, help='ReduceLROnPlateau 耐心轮数')
    parser.add_argument('--cosine_tmax', type=int, default=50, help='CosineAnnealingLR T_max')
    parser.add_argument('--cosine_eta_min', type=float, default=1e-6, help='CosineAnnealingLR 最低学习率')
    # EarlyStopping
    parser.add_argument('--min_delta', type=float, default=1e-4, help='EarlyStopping 最小改进阈值')
    parser.add_argument('--patience', type=int, default=7, help='EarlyStopping 耐心值（轮）')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='线性 warmup 的 epoch 数')
    parser.add_argument('--val_viz_num', type=int, default=3, help='每轮上传至 WandB 的验证样本数量')
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoader 并行 worker 数量')
    # debug
    parser.add_argument('--max_samples', type=int, default=10, help='最大加载样本数（快速验证）')
    # distributed
    parser.add_argument('--local_rank', type=int, default=0, help='分布式训练本地 GPU 编号')
    parser.add_argument('--dist', action='store_true', help='是否启用分布式训练')
    # wandb
    parser.add_argument('--entity', default='joint_angle', help='WandB 实体名（用户名或团队）')

    parser.add_argument('--device', default=('cuda' if torch.cuda.is_available() else 'cpu'), help='运行设备')
    return parser.parse_args()

def parse_args_r():
    parser = argparse.ArgumentParser(description="Seg→Pose Distill Training for B200 (512×512)")

    # 数据与输出
    parser.add_argument('--data_dir',      default='run/data', help='COCO 数据集根目录')
    parser.add_argument('--output_dir',    default='run',      help='输出文件目录')

    # 训练轮数
    parser.add_argument('--teacher_epochs', type=int, default=300, help='教师模型训练轮数')
    parser.add_argument('--student_epochs', type=int, default=100, help='学生蒸馏训练轮数')

    # 大分辨率下的 batch size
    parser.add_argument('--batch_size',    type=int, default=16,   help='训练批大小 (512×512 建议 8–16)')

    # 学习率
    parser.add_argument('--lr',            type=float, default=1e-4, help='初始学习率')

    # 学习率调度器参数
    parser.add_argument('--scheduler',      choices=['plateau','cosine'], default='plateau',
                        help='学习率调度策略')
    parser.add_argument('--plateau_factor', type=float, default=0.5,   help='ReduceLROnPlateau 因子')
    parser.add_argument('--plateau_patience',type=int,   default=5,   help='ReduceLROnPlateau 耐心轮数')
    parser.add_argument('--cosine_tmax',     type=int,    default=100, help='CosineAnnealingLR T_max')
    parser.add_argument('--cosine_eta_min',  type=float,  default=1e-6,help='CosineAnnealingLR 最低学习率')

    # EarlyStopping
    parser.add_argument('--min_delta',      type=float, default=1e-4, help='EarlyStopping 最小改进阈值')
    parser.add_argument('--patience',       type=int,   default=10,   help='EarlyStopping 耐心值（轮）')

    # Warmup & 可视化
    parser.add_argument('--warmup_epochs',  type=int, default=10, help='线性 warmup 的 epoch 数')
    parser.add_argument('--val_viz_num',    type=int, default=2,  help='每轮上传至 WandB 的验证样本数量')

    # DataLoader 并行
    parser.add_argument('--num_workers',    type=int, default=32, help='DataLoader 并行 worker 数量')

    # 调试用
    parser.add_argument('--max_samples',    type=int, default=None, help='最大加载样本数（快速验证）')

    # 分布式训练
    parser.add_argument('--local_rank',     type=int, default=0,    help='分布式训练本地 GPU 编号')
    parser.add_argument('--dist',           action='store_true',    help='是否启用分布式训练')

    # WandB
    parser.add_argument('--entity',         default='joint_angle', help='WandB 实体名（用户名或团队）')
    parser.add_argument('--device',         default=('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='运行设备')

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
    if args.scheduler=='plateau':
        sched_t = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_t, mode='min', factor=args.plateau_factor, patience=args.plateau_patience)
    else:
        sched_t = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_t, T_max=args.cosine_tmax, eta_min=args.cosine_eta_min)
    es_t = {'best_loss': float('inf'), 'counter': 0}
    # Student
    student = StudentModel().to(device)
    opt_s = torch.optim.Adam(student.parameters(), lr=args.lr)
    if args.scheduler=='plateau':
        sched_s = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_s, mode='min', factor=args.plateau_factor, patience=args.plateau_patience)
    else:
        sched_s = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_s, T_max=args.cosine_tmax, eta_min=args.cosine_eta_min)
    es_s = {'best_loss': float('inf'), 'counter': 0}
    return teacher,opt_t,sched_t,es_t,student,opt_s,sched_s,es_s


def extract_coords_from_heatmaps(heatmaps):
    B,K,H,W = heatmaps.shape
    coords = torch.zeros((B,K,2), device=heatmaps.device)
    for b in range(B):
        for k in range(K):
            idx = torch.argmax(heatmaps[b,k])
            y,x = divmod(idx.item(), W)
            coords[b,k] = torch.tensor([x*256/W, y*256/H], device=heatmaps.device)
    return coords


def visualize_batch(imgs, masks,
                    t_seg, s_seg,
                    t_kps, s_kps,
                    kps_list, vis_list, n):
    outputs=[]
    B=imgs.size(0)
    for i in range(min(n,B)):
        img_np=(imgs[i].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        def overlay(mt):
            m=(mt.cpu().numpy().squeeze()>0.5).astype(np.uint8)*255
            mbgr=cv2.cvtColor(m,cv2.COLOR_GRAY2BGR)
            return cv2.addWeighted(img_np,0.7,mbgr,0.3,0)
        entry={
            'GT_Segmentation':overlay(masks[i,0]),
            'Teacher_Segmentation':overlay(torch.sigmoid(t_seg[i,0]))
        }
        gt_kps=kps_list[i].reshape(-1,2); gt_vis=vis_list[i]
        entry['GT_Keypoints']=draw_keypoints_linked(img_np,gt_kps,gt_vis)
        t_vis=np.ones((t_kps.size(1),),int)
        entry['Teacher_Keypoints']=draw_keypoints_linked(img_np,t_kps[i].cpu().numpy(),t_vis)
        if s_seg is not None and s_kps is not None:
            entry['Student_Segmentation']=overlay(torch.sigmoid(s_seg[i,0]))
            s_vis=np.ones((s_kps.size(1),),int)
            entry['Student_Keypoints']=draw_keypoints_linked(img_np,s_kps[i].cpu().numpy(),s_vis)
        outputs.append(entry)
    return outputs


def train_one_epoch(model, loader, losses, optimizer=None, teacher=None, scaler=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    if teacher: teacher.eval()

    agg = {'seg': 0.0, 'pose': 0.0, 'distill': 0.0}
    tot = 0
    start = time.time()

    # 选择上下文
    ctx = torch.enable_grad() if training else torch.no_grad()

    for batch_idx, (imgs, masks, kps, vis, _, _) in enumerate(tqdm(loader, desc=('Training' if training else 'Evaluating'), leave=False)):

        imgs, masks = imgs.to(device), masks.to(device)
        kps = [t.to(device) for t in kps]
        vis = [t.to(device) for t in vis]

        with ctx:
            if training and scaler:
                # 混合精度前向/反向
                with torch.cuda.amp.autocast():
                    if teacher is None:
                        seg_out, heat, _ = model(imgs)
                        l_seg = losses['seg'](seg_out, masks)
                        l_pose = losses['pose'](heat, kps, vis)
                        loss = l_seg + l_pose
                        l_dist = 0
                    else:
                        with torch.no_grad():
                            t_seg, t_heat, _ = teacher(imgs)
                            tk = extract_coords_from_heatmaps(t_heat)
                        s_seg, sk = model(imgs)
                        loss, met = losses['distill']((s_seg, sk), (t_seg, tk), (masks, kps, vis))
                        l_seg = met['task_seg_loss']
                        l_pose = met['task_pose_loss']
                        l_dist = met['seg_distill_loss'] + met['pose_distill_loss']
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # 普通精度
                if teacher is None:
                    seg_out, heat, _ = model(imgs)
                    l_seg = losses['seg'](seg_out, masks)
                    l_pose = losses['pose'](heat, kps, vis)
                    loss = l_seg + l_pose
                    l_dist = 0
                else:
                    with torch.no_grad():
                        t_seg, t_heat, _ = teacher(imgs)
                        tk = extract_coords_from_heatmaps(t_heat)
                    s_seg, sk = model(imgs)
                    loss, met = losses['distill']((s_seg, sk), (t_seg, tk), (masks, kps, vis))
                    l_seg = met['task_seg_loss']
                    l_pose = met['task_pose_loss']
                    l_dist = met['seg_distill_loss'] + met['pose_distill_loss']
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        # 累计指标
        b = imgs.size(0)
        agg['seg'] += l_seg * b
        agg['pose'] += l_pose * b
        agg['distill'] += (l_dist if teacher else 0) * b
        tot += b

    if not training:
        torch.cuda.empty_cache()

    elapsed = time.time() - start
    return {k: v / tot for k, v in agg.items()}, elapsed


if __name__=='__main__':
    args = parse_args()
    print("======== Training Configuration ========")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("========================================")
    world_size, rank = setup_distributed(args)
    device = torch.device('cuda', args.local_rank) if args.dist else torch.device(args.device)

    # Create output dirs
    os.makedirs(args.output_dir, exist_ok=True)
    for sub in ['vis','models']:
        os.makedirs(os.path.join(args.output_dir, sub), exist_ok=True)

    # Data
    sampler = DistributedSampler if args.dist else lambda ds: None
    train_s = prepare_coco_dataset(args.data_dir, 'train', max_samples=args.max_samples)
    val_s   = prepare_coco_dataset(args.data_dir, 'val',   max_samples=args.max_samples)

    train_ds = COCODataset(train_s)
    val_ds   = COCODataset(val_s)
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

    # Models & DDP
    teacher, opt_t, sched_t, es_t, student, opt_s, sched_s, es_s = get_models_and_optim(args, device)
    if args.dist:
        teacher = DDP(teacher, device_ids=[args.local_rank])
        student = DDP(student, device_ids=[args.local_rank])

    # Losses & metrics.py
    seg_fn = SegmentationLoss().to(device)
    kp_fn  = KeypointsLoss().to(device)
    dist_fn = DistillationLoss().to(device)

    # Wandb
    logger_t = WandbLogger("Teacher", args.entity, config=vars(args))
    try:
        # ===== Teacher Training =====
        for epoch in trange(1, args.teacher_epochs+1, desc='Teacher'):
            # warmup
            if epoch <= args.warmup_epochs:
                lr = args.lr * epoch / args.warmup_epochs
                for g in opt_t.param_groups: g['lr'] = lr

            tr_metrics, tr_time = train_one_epoch(teacher, train_ld,
                                                  {'seg':seg_fn, 'pose':kp_fn},
                                                  optimizer=opt_t, teacher=None)
            val_metrics, _ = train_one_epoch(teacher, val_ld,
                                             {'seg':seg_fn, 'pose':kp_fn},
                                             optimizer=None, teacher=None)
            val_loss = val_metrics['seg'] + val_metrics['pose']

            # compute additional metrics.py
            miou = compute_miou(teacher, val_ld, device)
            pck  = compute_pck(teacher, val_ld, device)

            # scheduler
            if args.scheduler=='plateau': sched_t.step(val_loss)
            else: sched_t.step()

            # log scalars
            logger_t.log({
                'epoch': epoch,
                'train/seg_loss': tr_metrics['seg'],
                'train/pose_loss': tr_metrics['pose'],
                'val/loss': val_loss,
                'val/miou': miou,
                'val/pck': pck,
                'lr': opt_t.param_groups[0]['lr'],
                'time/epoch': tr_time,
                'hardware/gpu_mem_MB': torch.cuda.max_memory_allocated()/1024**2
            }, step=epoch)

            # log histograms
            for name, param in teacher.named_parameters():
                logger_t.log({f'parameters/{name}_hist': param.detach().cpu().numpy()}, step=epoch)
                if param.grad is not None:
                    logger_t.log({f'parameters/{name}_grad_hist': param.grad.detach().cpu().numpy()}, step=epoch)

            # visualization GT vs Teacher
            imgs, masks, kps, vis, _, _ = next(iter(val_ld))
            with torch.no_grad():
                t_seg_out, t_heat, _ = teacher(imgs.to(device))
                t_kps_pred = extract_coords_from_heatmaps(t_heat)
            vizs = visualize_batch(imgs, masks,
                                   t_seg_out, None,
                                   t_kps_pred, None,
                                   kps, vis,
                                   args.val_viz_num)
            for i, d in enumerate(vizs):
                logger_t.log({
                    f'viz/teacher_{i}/GT_Segmentation':      wandb.Image(d['GT_Segmentation']),
                    f'viz/teacher_{i}/Teacher_Segmentation': wandb.Image(d['Teacher_Segmentation']),
                    f'viz/teacher_{i}/GT_Keypoints':         wandb.Image(d['GT_Keypoints']),
                    f'viz/teacher_{i}/Teacher_Keypoints':    wandb.Image(d['Teacher_Keypoints'])
                }, step=epoch)

            # EarlyStopping
            if val_loss + args.min_delta < es_t['best_loss']:
                es_t['best_loss'] = val_loss
                es_t['counter'] = 0
                best_t = teacher.state_dict()
            else:
                es_t['counter'] += 1
                if es_t['counter'] >= args.patience:
                    logging.info(f"Teacher early stopping at epoch {epoch}")
                    break

        torch.save(best_t, os.path.join(args.output_dir, 'models', 'teacher.pth'))
        logger_t.finish()

        # ===== Student Distillation =====
        logger_s = WandbLogger("Student", args.entity, config=vars(args))

        teacher.load_state_dict(best_t)
        teacher.eval()
        for epoch in trange(1, args.student_epochs+1, desc='Student'):
            if epoch <= args.warmup_epochs:
                lr = args.lr * epoch / args.warmup_epochs
                for g in opt_s.param_groups: g['lr'] = lr

            tr_metrics_s, tr_time_s = train_one_epoch(student, train_ld,
                                                      {'distill':dist_fn},
                                                      optimizer=opt_s, teacher=teacher)
            val_metrics_s, _ = train_one_epoch(student, val_ld,
                                               {'distill':dist_fn},
                                               optimizer=None, teacher=teacher)
            val_loss_s = val_metrics_s['distill']

            # scheduler
            if args.scheduler=='plateau': sched_s.step(val_loss_s)
            else: sched_s.step()

            # compute metrics.py
            miou_s = compute_miou(student, val_ld, device)
            pck_s  = compute_pck(student, val_ld, device)

            logger_s.log({
                'epoch': epoch,
                'train/distill_loss': tr_metrics_s['distill'],
                'val/distill_loss': val_loss_s,
                'val/miou': miou_s,
                'val/pck': pck_s,
                'lr': opt_s.param_groups[0]['lr'],
                'time/epoch': tr_time_s,
                'hardware/gpu_mem_MB': torch.cuda.max_memory_allocated()/1024**2
            }, step=epoch)

            # histograms
            for name, param in student.named_parameters():
                logger_s.log({f'student/{name}_hist': param.detach().cpu().numpy()}, step=epoch)
                if param.grad is not None:
                    logger_s.log({f'student/{name}_grad_hist': param.grad.detach().cpu().numpy()}, step=epoch)

            # visualization GT vs Teacher vs Student
            imgs, masks, kps, vis, _, _ = next(iter(val_ld))
            with torch.no_grad():
                t_seg_out, t_heat, _ = teacher(imgs.to(device))
                t_kps_pred = extract_coords_from_heatmaps(t_heat)
                s_seg_out, s_kps_pred = student(imgs.to(device))
            vizs = visualize_batch(imgs, masks,
                                   t_seg_out, s_seg_out,
                                   t_kps_pred, s_kps_pred,
                                   kps, vis,
                                   args.val_viz_num)
            for i, d in enumerate(vizs):
                logger_s.log({
                    f'viz/student_{i}/GT_Segmentation':      wandb.Image(d['GT_Segmentation']),
                    f'viz/student_{i}/Teacher_Segmentation': wandb.Image(d['Teacher_Segmentation']),
                    f'viz/student_{i}/Student_Segmentation': wandb.Image(d['Student_Segmentation']),
                    f'viz/student_{i}/GT_Keypoints':         wandb.Image(d['GT_Keypoints']),
                    f'viz/student_{i}/Teacher_Keypoints':    wandb.Image(d['Teacher_Keypoints']),
                    f'viz/student_{i}/Student_Keypoints':    wandb.Image(d['Student_Keypoints'])
                }, step=epoch)

            if val_loss_s + args.min_delta < es_s['best_loss']:
                es_s['best_loss'] = val_loss_s
                es_s['counter'] = 0
                best_s = student.state_dict()
            else:
                es_s['counter'] += 1
                if es_s['counter'] >= args.patience:
                    logging.info(f"Student early stopping at epoch {epoch}")
                    break

        torch.save(best_s, os.path.join(args.output_dir, 'models', 'student.pth'))
        logger_s.finish()

    except Exception:
        logging.exception("Training failed at rank %d", rank)
        raise
    finally:
        if args.dist:
            dist.destroy_process_group()