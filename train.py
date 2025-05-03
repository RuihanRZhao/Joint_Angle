# train.py: 教师模型训练与学生模型蒸馏训练脚本（含更多训练监控信息与 WandB 上传）
import os
import argparse
import random
import time
import torch
import wandb
from tqdm import tqdm
from tqdm import trange
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from models.teacher_model import TeacherModel
from models.student_model import StudentModel
from models.segmentation_model import UNetSegmentation
from models.pose_model import PoseEstimationModel

from utils.coco import prepare_coco_dataset, COCODataset, collate_fn
from utils.visualization import visualize_raw_samples, visualize_predictions
from utils.loss import SegmentationLoss, KeypointsLoss, DistillationLoss
from utils.wandbLogger import WandbLogger


def parse_args():
    parser = argparse.ArgumentParser(description="教师模型训练与学生蒸馏训练脚本")
    parser.add_argument('--data_dir', type=str, default='run/data', help='COCO 数据集根目录')
    parser.add_argument('--output_dir', type=str, default='run', help='输出文件目录')
    parser.add_argument('--teacher_epochs', type=int, default=2, help='教师模型训练轮数')
    parser.add_argument('--student_epochs', type=int, default=2, help='学生蒸馏训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='训练批大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='初始学习率')
    parser.add_argument('--min_delta', type=float, default=1e-4, help='EarlyStopping 最小改进阈值')
    parser.add_argument('--patience', type=int, default=7, help='EarlyStopping 耐心值（轮）')
    parser.add_argument('--scheduler', type=str, default='plateau', choices=['plateau', 'cosine'], help='学习率调度策略')
    parser.add_argument('--device', type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'), help='运行设备')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='线性 warmup 的 epoch 数')
    parser.add_argument('--val_viz_num', type=int, default=3, help='每轮上传至 WandB 的验证样本数量')
    parser.add_argument('--num_workers', type=int, default=32, help='DataLoader 的并行 worker 数量')
    return parser.parse_args()


def parse_args_r():
    parser = argparse.ArgumentParser(description="教师模型训练与学生蒸馏训练脚本")
    parser.add_argument('--data_dir', type=str, default='run/data', help='COCO 数据集根目录')
    parser.add_argument('--output_dir', type=str, default='run', help='输出文件目录')
    parser.add_argument('--teacher_epochs', type=int, default=300, help='教师模型训练轮数')
    parser.add_argument('--student_epochs', type=int, default=100, help='学生蒸馏训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='训练批大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='初始学习率')
    parser.add_argument('--min_delta', type=float, default=1e-5, help='EarlyStopping 最小改进阈值')
    parser.add_argument('--patience', type=int, default=15, help='EarlyStopping 耐心值（轮）')
    parser.add_argument('--scheduler', type=str, default='plateau', choices=['plateau', 'cosine'], help='学习率调度策略')
    parser.add_argument('--device', type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'), help='运行设备')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='线性 warmup 的 epoch 数')
    parser.add_argument('--val_viz_num', type=int, default=5, help='每轮上传至 WandB 的验证样本数量')
    parser.add_argument('--num_workers', type=int, default=32, help='DataLoader 的并行 worker 数量')
    return parser.parse_args()


class EarlyStopping:
    """
    监控验证损失，当连续若干 epoch 无改进时提前终止训练
    """
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def step(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping: {self.counter}/{self.patience} (val_loss not improving)")
            if self.counter >= self.patience:
                print("EarlyStopping: Stop training!")
                self.early_stop = True

def upsample_mask(mask_arr_256: np.ndarray, target_size: tuple) -> np.ndarray:
    """
    将 256×256 的 mask 最近邻地 upsample 回原图尺寸。
    """
    return np.array(
        Image.fromarray(mask_arr_256).resize(target_size, resample=Image.NEAREST)
    )

def draw_segmentation(img_arr, mask_arr):
    """
    在原图上叠加绿色半透明分割掩码，返回 RGB 图像
    """
    vis = Image.fromarray(img_arr).convert('RGBA')
    draw = ImageDraw.Draw(vis, 'RGBA')
    m = Image.fromarray(mask_arr).convert('L')
    draw.bitmap((0, 0), m, fill=(0, 255, 0, 100))
    return vis.convert('RGB')

def draw_keypoints(img_arr, kps, vis):
    """
    在原图上画红色关键点，返回 RGB 图像
    """
    vis_img = Image.fromarray(img_arr).convert('RGBA')
    draw   = ImageDraw.Draw(vis_img, 'RGBA')
    h, w = img_arr.shape[:2]
    for (x_raw, y_raw), v in zip(kps.numpy(), vis.numpy()):
        if v > 0:
            x = int(round(float(x_raw)))
            y = int(round(float(y_raw)))
            if 0 <= x < w and 0 <= y < h:
                draw.ellipse((x-3, y-3, x+3, y+3), fill=(255, 0, 0, 255))
    return vis_img.convert('RGB')

def log_gt_and_teacher_to_wandb(
    dataloader, model, device, wandb_logger, num_viz: int, epoch: int
):
    """
    在原始分辨率上可视化 GT vs Teacher，上传 WandB。
    """
    model.eval()
    dataset = dataloader.dataset

    for idx in random.sample(range(len(dataset)), num_viz):
        # unpack the 6-tuple: (img_tensor, gt_mask, kps_res, vis, path, kps_orig)
        img_tensor, gt_mask_tensor, kps_res, vis_tensor, path, kps_orig = dataset[idx]

        # 1) load full-res image
        orig = Image.open(path).convert('RGB')
        img_arr_orig = np.array(orig)

        # 2) upsample GT mask
        gt_mask_256 = (gt_mask_tensor.numpy().squeeze() * 255).astype('uint8')
        gt_mask_up  = upsample_mask(gt_mask_256, orig.size)

        # 3) run teacher prediction & upsample
        inp = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(inp)
        pred = out[0] if isinstance(out, tuple) else out
        t_mask_256 = (pred.cpu().numpy().squeeze() * 255).astype('uint8')
        t_mask_up  = upsample_mask(t_mask_256, orig.size)

        # 4) draw on full-res
        gt_seg = draw_segmentation(img_arr_orig, gt_mask_up)
        gt_kp  = draw_keypoints(   img_arr_orig, kps_orig,    vis_tensor)
        t_seg  = draw_segmentation(img_arr_orig, t_mask_up)
        t_kp   = draw_keypoints(   img_arr_orig, kps_orig,    vis_tensor)

        # 5) upload to WandB
        wandb_logger.log({
            f"viz/{idx}/GT_Segmentation": wandb.Image(gt_seg),
            f"viz/{idx}/GT_Keypoints":   wandb.Image(gt_kp),
            f"viz/{idx}/Teacher_Seg":     wandb.Image(t_seg),
            f"viz/{idx}/Teacher_KP":      wandb.Image(t_kp),
        }, step=epoch)

    model.train()

def log_gt_teacher_student_to_wandb(
    dataloader, teacher, student, device,
    wandb_logger, num_viz: int, epoch: int
):
    """
    在原始分辨率上可视化 GT vs Teacher vs Student，上传 WandB。
    """
    teacher.eval(); student.eval()
    dataset = dataloader.dataset

    for idx in random.sample(range(len(dataset)), num_viz):
        img_tensor, gt_mask_tensor, kps_res, vis_tensor, path, kps_orig = dataset[idx]
        orig = Image.open(path).convert('RGB')
        img_arr_orig = np.array(orig)

        # upsample GT
        gt_mask_up = upsample_mask(
            (gt_mask_tensor.numpy().squeeze() * 255).astype('uint8'),
            orig.size
        )

        # teacher & student preds
        inp = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            t_out = teacher(inp)
            s_out = student(inp)
        t_pred = t_out[0] if isinstance(t_out, tuple) else t_out
        s_pred = s_out[0] if isinstance(s_out, tuple) else s_out

        # upsample masks
        t_mask_up = upsample_mask(
            (t_pred.cpu().numpy().squeeze() * 255).astype('uint8'),
            orig.size
        )
        s_mask_up = upsample_mask(
            (s_pred.cpu().numpy().squeeze() * 255).astype('uint8'),
            orig.size
        )

        # draw all six views
        images = {
            f"viz/{idx}/GT_Seg":      draw_segmentation(img_arr_orig,  gt_mask_up),
            f"viz/{idx}/GT_KP":       draw_keypoints(img_arr_orig,    kps_orig,    vis_tensor),
            f"viz/{idx}/Teacher_Seg": draw_segmentation(img_arr_orig,  t_mask_up),
            f"viz/{idx}/Teacher_KP":  draw_keypoints(img_arr_orig,    kps_orig,    vis_tensor),
            f"viz/{idx}/Student_Seg": draw_segmentation(img_arr_orig,  s_mask_up),
            f"viz/{idx}/Student_KP":  draw_keypoints(img_arr_orig,    kps_orig,    vis_tensor),
        }

        wandb_logger.log({k: wandb.Image(v) for k, v in images.items()}, step=epoch)

    teacher.train(); student.train()


def train_one_epoch_teacher(
    model, loader, optimizer, seg_loss_fn, kp_loss_fn, device
):
    """
    教师模型单 epoch 训练，注意 loader 解包现在是 6 个元素。
    """
    model.train()
    total_loss = 0.0
    batch_times, grad_norms, weight_norms = [], [], []

    for imgs, masks, kps, vs, _, _ in tqdm(loader, desc="Training", leave=False):
        start_t = time.time()

        imgs, masks = imgs.to(device), masks.to(device)
        kps = [t.to(device) for t in kps]
        vs  = [t.to(device) for t in vs]

        seg_logits, pose_logits = model(imgs)
        loss = seg_loss_fn(seg_logits, masks) + kp_loss_fn(pose_logits, kps, vs)

        optimizer.zero_grad()
        loss.backward()

        total_norm = torch.norm(torch.stack([
            torch.norm(p.grad.detach())
            for p in model.parameters() if p.grad is not None
        ]))
        grad_norms.append(total_norm.item())

        weight_norm = torch.norm(torch.stack([
            torch.norm(p.detach())
            for p in model.parameters()
        ]))
        weight_norms.append(weight_norm.item())

        optimizer.step()
        batch_times.append(time.time() - start_t)
        total_loss += loss.detach().item()

    avg_loss = total_loss / len(loader)
    return avg_loss, np.mean(batch_times), np.mean(grad_norms), np.mean(weight_norms)



def evaluate_teacher(
    model, loader, seg_loss_fn, kp_loss_fn, device
):
    """
    教师模型单 epoch 验证，loader 解包 6 个元素。
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for imgs, masks, kps, vs, _, _ in tqdm(loader, desc="Evaluating", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            kps = [t.to(device) for t in kps]
            vs  = [t.to(device) for t in vs]

            seg_logits, pose_logits = model(imgs)
            total_loss += (
                seg_loss_fn(seg_logits, masks)
                + kp_loss_fn(pose_logits, kps, vs)
            ).item()

    return total_loss / len(loader)


def train_one_epoch_student(
    student, teacher, loader, optimizer, distill_loss_fn, device
):
    """
    学生模型蒸馏单 epoch 训练，loader 解包 6 个元素。
    """
    student.train()
    teacher.eval()
    total_loss = 0.0
    batch_times, grad_norms, weight_norms = [], [], []
    metrics_agg = {}

    for imgs, masks, kps, vs, _, _ in tqdm(loader, desc="Training", leave=False):
        start_t = time.time()

        imgs, masks = imgs.to(device), masks.to(device)
        kps = [t.to(device) for t in kps]
        vs  = [t.to(device) for t in vs]

        with torch.no_grad():
            t_seg, t_kp = teacher(imgs)
        s_seg, s_kp = student(imgs)

        loss, metrics = distill_loss_fn(
            (s_seg, s_kp), (t_seg, t_kp), (masks, kps, vs)
        )

        optimizer.zero_grad()
        loss.backward()

        total_norm = torch.norm(torch.stack([
            torch.norm(p.grad.detach())
            for p in student.parameters() if p.grad is not None
        ]))
        grad_norms.append(total_norm.item())

        weight_norm = torch.norm(torch.stack([
            torch.norm(p.detach())
            for p in student.parameters()
        ]))
        weight_norms.append(weight_norm.item())

        optimizer.step()
        batch_times.append(time.time() - start_t)
        total_loss += loss.detach().item()

        for k, v in metrics.items():
            metrics_agg[k] = metrics_agg.get(k, 0.0) + v

    avg_metrics = {k: v / len(loader) for k, v in metrics_agg.items()}
    return (total_loss / len(loader),
            np.mean(batch_times),
            np.mean(grad_norms),
            np.mean(weight_norms),
            avg_metrics)


def evaluate_student(
    student, teacher, loader, seg_loss_fn, kp_loss_fn, device
):
    """
    学生模型蒸馏单 epoch 验证，loader 解包 6 个元素。
    """
    student.eval()
    total_loss = 0.0

    with torch.no_grad():
        for imgs, masks, kps, vs, _, _ in tqdm(loader, desc="Evaluating", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            kps = [t.to(device) for t in kps]
            vs  = [t.to(device) for t in vs]

            s_seg, s_kp = student(imgs)
            total_loss += (
                seg_loss_fn(s_seg, masks)
                + kp_loss_fn(s_kp, kps, vs)
            ).item()

    return total_loss / len(loader)



def main():
    args = parse_args()
    base_lr = args.lr
    print(f"Using device: {args.device}")
    os.makedirs(args.output_dir, exist_ok=True)
    for sub in ['data', 'vis', 'models']:
        os.makedirs(os.path.join(args.output_dir, sub), exist_ok=True)

    train_samples = prepare_coco_dataset(args.data_dir, 'train')
    val_samples = prepare_coco_dataset(args.data_dir, 'val')
    visualize_raw_samples(train_samples[:5], os.path.join(args.output_dir, 'vis', 'raw'))

    train_loader = DataLoader(
            COCODataset(train_samples),
            batch_size = args.batch_size,
        shuffle = True,
        collate_fn = collate_fn,
        num_workers = args.num_workers,
        pin_memory = True,
    )
    val_loader = DataLoader(
            COCODataset(val_samples),
            batch_size = 1,
        shuffle = False,
        collate_fn = collate_fn,
        num_workers = args.num_workers,
        pin_memory = True,
    )

    device = torch.device(args.device)

    # ------------------ 教师模型训练 ------------------
    teacher = TeacherModel()
    teacher.segmentation = UNetSegmentation(in_channels=3, num_classes=1)
    teacher.pose = PoseEstimationModel(in_channels=4)
    teacher.to(device)
    optimizer_t = torch.optim.Adam(teacher.parameters(), lr=args.lr)
    scheduler_t = ReduceLROnPlateau(optimizer_t, mode='min', factor=0.5, patience=3) if args.scheduler=='plateau' else CosineAnnealingLR(optimizer_t, T_max=args.teacher_epochs)
    earlystop_t = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
    wandb_logger_t = WandbLogger(_project="Teacher_Model", _entity="joint_angle", config=vars(args))

    # 参数数量
    total_params = sum(p.numel() for p in teacher.parameters())
    wandb_logger_t.log({'teacher/num_parameters': total_params}, step=0)

    for epoch in trange(1, args.teacher_epochs+1, desc="Epochs"):
        # —— Linear Warmup ——
        if epoch <= args.warmup_epochs:
            warm_lr = base_lr * epoch / args.warmup_epochs
            for g in optimizer_t.param_groups:
                g['lr'] = warm_lr

        epoch_start = time.time()
        train_loss, train_step_time, train_grad_norm, train_weight_norm = train_one_epoch_teacher(
            teacher, train_loader, torch.optim.Adam(teacher.parameters(), lr=args.lr),
            SegmentationLoss(), KeypointsLoss(), device)
        val_loss = evaluate_teacher(teacher, val_loader, SegmentationLoss(), KeypointsLoss(), device)
        epoch_time = time.time() - epoch_start

        # GPU 内存使用
        mem_alloc = torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0
        mem_reserved = torch.cuda.max_memory_reserved(device) if device.type=='cuda' else 0

        # 日志上传
        wandb_logger_t.log({
            'epoch': epoch,
            'loss/train_loss': train_loss,
            'loss/val_loss': val_loss,
            'teacher/lr': optimizer_t.param_groups[0]['lr'],
            'time/avg_train_step_time': train_step_time,
            'time/epoch_time': epoch_time,
            'teacher/avg_grad_norm': train_grad_norm,
            'teacher/avg_weight_norm': train_weight_norm,
            'hardware/throughput(samples/sec)': args.batch_size / train_step_time,
            'hardware/gpu_memory_allocated': mem_alloc,
            'hardware/gpu_memory_reserved': mem_reserved
        }, step=epoch)

        # 参数与梯度分布
        hist_logs = {}
        for name, param in teacher.named_parameters():
            hist_logs[f"teacher/{name}_weight"] = wandb.Histogram(param.detach().cpu().numpy())
            if param.grad is not None:
                hist_logs[f"teacher/{name}_grad"] = wandb.Histogram(param.grad.detach().cpu().numpy())
        wandb_logger_t.log(hist_logs, step=epoch)

        # 上传 GT vs Teacher 对比图
        log_gt_and_teacher_to_wandb(
            dataloader=val_loader,
            model=teacher,
            device=device,
            wandb_logger=wandb_logger_t,
            num_viz=args.val_viz_num,
            epoch=epoch,
        )

        if earlystop_t.step(val_loss) or earlystop_t.early_stop:
            break

    torch.save(teacher.state_dict(), os.path.join(args.output_dir, 'models', 'teacher.pth'))
    wandb_logger_t.finish()
    print("教师模型训练完成，权重已保存。")

    # ------------------ 学生模型蒸馏训练 ------------------
    student = StudentModel(num_keypoints=17, seg_channels=1).to(device)
    optimizer_s = torch.optim.Adam(student.parameters(), lr=args.lr)
    scheduler_s = ReduceLROnPlateau(optimizer_s, mode='min', factor=0.5, patience=3) if args.scheduler=='plateau' else CosineAnnealingLR(optimizer_s, T_max=args.student_epochs)
    earlystop_s = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
    wandb_logger_s = WandbLogger(_project="Student_Model", _entity="joint_angle", config=vars(args))

    total_params_s = sum(p.numel() for p in student.parameters())
    wandb_logger_s.log({'student/num_parameters': total_params_s}, step=0)

    for epoch in trange(1, args.student_epochs+1, desc="Epochs"):
        # —— Linear Warmup ——
        if epoch <= args.warmup_epochs:
            warm_lr = base_lr * epoch / args.warmup_epochs
            for g in optimizer_t.param_groups:
                g['lr'] = warm_lr
        epoch_start = time.time()
        train_loss_s, student_step_time, student_grad_norm, student_weight_norm, metrics_s = train_one_epoch_student(
            student, teacher, train_loader, optimizer_s, DistillationLoss(), device)
        val_loss_s = evaluate_student(student, teacher, val_loader, SegmentationLoss(), KeypointsLoss(), device)
        epoch_time_s = time.time() - epoch_start

        mem_alloc_s = torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0
        mem_reserved_s = torch.cuda.max_memory_reserved(device) if device.type=='cuda' else 0

        log_data_s = {
            'epoch': epoch,
            'loss/train_loss': train_loss_s,
            'loss/val_loss': val_loss_s,
            'student/lr': optimizer_s.param_groups[0]['lr'],
            'time/avg_train_step_time': student_step_time,
            'time/epoch_time': epoch_time_s,
            'student/avg_grad_norm': student_grad_norm,
            'student/avg_weight_norm': student_weight_norm,
            'hardware/throughput(samples/sec)': args.batch_size / student_step_time,
            'hardware/gpu_memory_allocated': mem_alloc_s,
            'hardware/gpu_memory_reserved': mem_reserved_s
        }
        log_data_s.update(metrics_s)
        wandb_logger_s.log(log_data_s, step=epoch)

        hist_logs_s = {}
        for name, param in student.named_parameters():
            hist_logs_s[f"student/{name}_weight"] = wandb.Histogram(param.detach().cpu().numpy())
            if param.grad is not None:
                hist_logs_s[f"student/{name}_grad"] = wandb.Histogram(param.grad.detach().cpu().numpy())
        wandb_logger_s.log(hist_logs_s, step=epoch)

        # 上传 GT vs Teacher vs Student 对比图
        log_gt_teacher_student_to_wandb(
            dataloader = val_loader,
            teacher = teacher,
            student = student,
            device = device,
            wandb_logger = wandb_logger_s,
            num_viz = args.val_viz_num,
            epoch = epoch,
        )

        if earlystop_s.step(val_loss_s) or earlystop_s.early_stop:
            break

    torch.save(student.state_dict(), os.path.join(args.output_dir, 'models', 'student.pth'))
    wandb_logger_s.finish()
    print("学生模型蒸馏训练完成，权重已保存。")


if __name__ == '__main__':
    main()
