import os
import argparse
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

# 模型和组件导入
from models.teacher_model import TeacherModel
from models.student_model import StudentModel
from models.segmentation_model import UNetSegmentation
from models.pose_model import PoseEstimationModel

# 工具模块
from utils.coco import prepare_coco_dataset, COCODataset, collate_fn
from utils.visualization import visualize_raw_samples, visualize_predictions
from utils.loss import SegmentationLoss, KeypointsLoss, DistillationLoss

# wandb 封装
from utils.wandbLogger import WandbLogger


# EarlyStopping 类：当验证损失连续多次无改善时提前终止&#8203;:contentReference[oaicite:6]{index=6}
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
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

def parse_args():
    parser = argparse.ArgumentParser(description="教师模型训练与学生蒸馏训练")
    parser.add_argument('--data_dir', type=str, default='run/data', help='COCO 数据集根目录')
    parser.add_argument('--output_dir', type=str, default='run', help='输出文件目录')
    parser.add_argument('--teacher_epochs', type=int, default=2, help='教师模型训练轮数')
    parser.add_argument('--student_epochs', type=int, default=2, help='学生蒸馏训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='初始学习率')
    parser.add_argument('--min_delta', type=float, default=1e-4, help='EarlyStopping 最小增益阈值')
    parser.add_argument('--patience', type=int, default=5, help='EarlyStopping 耐心值（轮）')
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine'], help='学习率调度策略')
    parser.add_argument('--device', type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='运行设备')
    args = parser.parse_args()
    return args


def train_one_epoch_teacher(model, loader, optimizer, seg_loss_fn, kp_loss_fn, device):
    model.train()
    total_loss = 0.0
    for imgs, masks, kps, vs, _ in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        kps = [t.to(device) for t in kps]
        vs = [t.to(device) for t in vs]
        seg_logits, pose_logits = model(imgs)
        loss = seg_loss_fn(seg_logits, masks) + kp_loss_fn(pose_logits, kps, vs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate_teacher(model, loader, seg_loss_fn, kp_loss_fn, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for imgs, masks, kps, vs, _ in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            kps = [t.to(device) for t in kps]
            vs = [t.to(device) for t in vs]
            seg_logits, pose_logits = model(imgs)
            loss = seg_loss_fn(seg_logits, masks) + kp_loss_fn(pose_logits, kps, vs)
            total_loss += loss.item()
    return total_loss / len(loader)


def train_one_epoch_student(student, teacher, loader, optimizer, distill_loss_fn, device):
    student.train()
    teacher.eval()
    total_loss = 0.0
    metrics_agg = {}
    for imgs, masks, kps, vs, _ in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        kps = [t.to(device) for t in kps]
        vs = [t.to(device) for t in vs]
        with torch.no_grad():
            t_seg, t_kp = teacher(imgs)
        s_seg, s_kp = student(imgs)
        loss, metrics = distill_loss_fn((s_seg, s_kp), (t_seg, t_kp), (masks, kps, vs))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # 累计蒸馏过程中的各种损失
        for k, v in metrics.items():
            metrics_agg[k] = metrics_agg.get(k, 0.0) + v
    # 取平均
    metrics_avg = {k: v/len(loader) for k, v in metrics_agg.items()}
    return total_loss / len(loader), metrics_avg


def evaluate_student(student, teacher, loader, seg_loss_fn, kp_loss_fn, device):
    student.eval()
    total_loss = 0.0
    with torch.no_grad():
        for imgs, masks, kps, vs, _ in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            kps = [t.to(device) for t in kps]
            vs = [t.to(device) for t in vs]
            s_seg, s_kp = student(imgs)
            loss = seg_loss_fn(s_seg, masks) + kp_loss_fn(s_kp, kps, vs)
            total_loss += loss.item()
    return total_loss / len(loader)


def infer_and_visualize(model, loader, device, outdir, prefix):
    model.eval()
    results = []
    vis_dir = os.path.join(outdir, 'vis', prefix)
    os.makedirs(vis_dir, exist_ok=True)
    for imgs, masks, kps, vs, paths in loader:
        imgs = imgs.to(device)
        with torch.no_grad():
            seg_logits, kp_logits = model(imgs)
        seg_prob = torch.sigmoid(seg_logits)
        # 恢复到原图尺寸
        seg_resized = torch.nn.functional.interpolate(seg_prob, size=masks.shape[-2:],
                                                      mode='bilinear', align_corners=False)
        seg_mask = (seg_resized > 0.5).squeeze(1).cpu().numpy()
        hm = kp_logits.cpu().numpy()
        for i, path in enumerate(paths):
            orig = Image.open(path)
            orig_w, orig_h = orig.size
            mask_pred = Image.fromarray((seg_mask[i] * 255).astype(np.uint8)).resize((orig_w, orig_h))
            mask_pred_np = np.array(mask_pred).astype(np.uint8)
            gt_mask_img = Image.fromarray((masks[i].squeeze(0).cpu().numpy()*255).astype(np.uint8))
            gt_mask_np = np.array(gt_mask_img.resize((orig_w, orig_h))).astype(np.uint8) > 128
            coords = []
            for heat in hm[i]:
                y, x = np.unravel_index(heat.argmax(), heat.shape)
                coord_x = x * orig_w / heat.shape[1]
                coord_y = y * orig_h / heat.shape[0]
                coords.append([coord_x, coord_y])
            results.append({
                'image_path': path,
                'mask': mask_pred_np,
                'keypoints': np.array(coords, dtype=np.float32),
                'visibility': np.ones((len(coords),), dtype=np.int32),
                'gt_mask': gt_mask_np.astype(np.uint8),
                'gt_keypoints': (kps[i].cpu().numpy().astype(np.float32) * np.array([[orig_w/256, orig_h/256]])),
                'gt_visibility': vs[i].cpu().numpy().astype(np.int32)
            })
    visualize_predictions(results, vis_dir)
    print(f"{prefix} 可视化结果保存在 {vis_dir}")


def main():
    args = parse_args()
    print("Running on device:", args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    for sub in ['data', 'vis', 'models']:
        os.makedirs(os.path.join(args.output_dir, sub), exist_ok=True)

    # 加载数据集（示例使用全部样本）
    train_samples = prepare_coco_dataset(args.data_dir, 'train')
    val_samples = prepare_coco_dataset(args.data_dir, 'validation')
    if not train_samples or not val_samples:
        raise RuntimeError("COCO 数据集加载失败，请检查路径")

    # 可视化部分原始样本
    visualize_raw_samples(train_samples[:5], os.path.join(args.output_dir, 'vis', 'raw'))

    # 数据迭代器（多 worker, pin_memory=True 提高吞吐量&#8203;:contentReference[oaicite:7]{index=7}）
    train_loader = DataLoader(
        dataset=COCODataset(train_samples),
        batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        dataset=COCODataset(val_samples),
        batch_size=4, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True
    )

    device = torch.device(args.device)

    # 初始化教师模型（Segmentation + Pose）
    teacher = TeacherModel()
    teacher.segmentation = UNetSegmentation(in_channels=3, num_classes=1)  # 例如二分类分割
    teacher.pose = PoseEstimationModel(in_channels=4)
    teacher.to(device)
    optimizer_t = torch.optim.Adam(teacher.parameters(), lr=args.lr)
    # 学习率调度器
    if args.scheduler == 'plateau':
        scheduler_t = ReduceLROnPlateau(optimizer_t, mode='min', factor=0.5, patience=3, verbose=True)
    else:
        scheduler_t = CosineAnnealingLR(optimizer_t, T_max=args.teacher_epochs)
    earlystop_t = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

    # wandb 初始化（教师模型）
    wandb_logger_t = WandbLogger(
        project_name="TeacherModel_Project_Placeholder",
        config={
            "epochs": args.teacher_epochs,
            "batch_size": args.batch_size,
            "optimizer": "Adam",
            "lr": args.lr,
            "scheduler": args.scheduler,
        }
    )

    print("开始训练教师模型...")
    seg_loss_fn = SegmentationLoss()
    kp_loss_fn = KeypointsLoss()
    for epoch in range(1, args.teacher_epochs + 1):
        loss_train = train_one_epoch_teacher(teacher, train_loader, optimizer_t, seg_loss_fn, kp_loss_fn, device)
        loss_val = evaluate_teacher(teacher, val_loader, seg_loss_fn, kp_loss_fn, device)
        print(f"[Teacher Epoch {epoch}] 训练损失={loss_train:.4f}, 验证损失={loss_val:.4f}")

        # 更新学习率调度器和 EarlyStopping
        if args.scheduler == 'plateau':
            scheduler_t.step(loss_val)  # 当验证损失停滞时自动调整&#8203;:contentReference[oaicite:8]{index=8}
        else:
            scheduler_t.step()
        earlystop_t.step(loss_val)
        # wandb 记录（每轮一次）
        wandb_logger_t.log({
            "epoch": epoch,
            "train_loss": loss_train,
            "val_loss": loss_val,
            "lr": optimizer_t.param_groups[0]['lr']
        }, step=epoch)

        if earlystop_t.early_stop:
            break
    # 保存教师模型权重
    torch.save(teacher.state_dict(), os.path.join(args.output_dir, 'models', 'teacher.pth'))
    wandb_logger_t.finish()
    print("教师模型训练完成，权重已保存。")

    # 初始化学生模型
    student = StudentModel(num_keypoints=17, seg_channels=1).to(device)
    optimizer_s = torch.optim.Adam(student.parameters(), lr=args.lr)
    if args.scheduler == 'plateau':
        scheduler_s = ReduceLROnPlateau(optimizer_s, mode='min', factor=0.5, patience=3, verbose=True)
    else:
        scheduler_s = CosineAnnealingLR(optimizer_s, T_max=args.student_epochs)
    earlystop_s = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

    # wandb 初始化（学生模型）
    wandb_logger_s = WandbLogger(
        project_name="StudentModel_Project_Placeholder",
        config={
            "epochs": args.student_epochs,
            "batch_size": args.batch_size,
            "optimizer": "Adam",
            "lr": args.lr,
            "scheduler": args.scheduler,
            "alpha": 0.5,
            "temperature": 2.0
        }
    )

    print("开始学生模型蒸馏训练...")
    distill_loss_fn = DistillationLoss(alpha=0.5, temperature=2.0)
    for epoch in range(1, args.student_epochs + 1):
        loss_train, metrics_train = train_one_epoch_student(
            student, teacher, train_loader, optimizer_s, distill_loss_fn, device
        )
        loss_val = evaluate_student(student, teacher, val_loader, seg_loss_fn, kp_loss_fn, device)
        print(f"[Student Epoch {epoch}] 训练总损失={loss_train:.4f}, 验证损失={loss_val:.4f}")

        # 更新学习率调度器和 EarlyStopping
        if args.scheduler == 'plateau':
            scheduler_s.step(loss_val)
        else:
            scheduler_s.step()
        earlystop_s.step(loss_val)
        # wandb 记录
        log_dict = {
            "epoch": epoch,
            "train_loss": loss_train,
            "val_loss": loss_val,
            "lr": optimizer_s.param_groups[0]['lr']
        }
        log_dict.update(metrics_train)  # 包含分割和姿态任务损失、蒸馏损失等
        wandb_logger_s.log(log_dict, step=epoch)

        if earlystop_s.early_stop:
            break
    # 保存学生模型权重
    torch.save(student.state_dict(), os.path.join(args.output_dir, 'models', 'student.pth'))
    wandb_logger_s.finish()
    print("学生模型蒸馏训练完成，权重已保存。")

    # 最终对比：将教师和学生分别应用于验证集并可视化结果
    print("开始推理与可视化...")
    infer_and_visualize(teacher, val_loader, device, args.output_dir, prefix='teacher')
    infer_and_visualize(student, val_loader, device, args.output_dir, prefix='student')
    print("训练与可视化结束，所有结果保存在 run/ 目录下。")


if __name__ == '__main__':
    main()
