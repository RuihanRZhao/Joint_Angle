# train.py: 教师模型训练与学生模型蒸馏训练脚本（含 WandB 验证集对比图上传）
import os  # 文件和路径操作
import argparse  # 命令行参数解析
import random  # 随机抽样
import torch  # PyTorch 核心库
import numpy as np  # 数值计算
from PIL import Image  # 图像读写
from torch.utils.data import DataLoader  # 数据加载
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR  # 学习率调度

# 模型和组件导入
from models.teacher_model import TeacherModel  # 教师模型类
from models.student_model import StudentModel  # 学生模型类
from models.segmentation_model import UNetSegmentation  # 分割子模块
from models.pose_model import PoseEstimationModel  # 姿态估计子模块

# 工具模块
from utils.coco import prepare_coco_dataset, COCODataset, collate_fn  # COCO 数据集加载与打包函数
from utils.visualization import visualize_raw_samples, visualize_predictions  # 可视化工具
from utils.loss import SegmentationLoss, KeypointsLoss, DistillationLoss  # 损失函数

# wandb 封装
from utils.wandbLogger import WandbLogger  # 封装后的 wandb 类
import wandb  # Weights & Biases

# ------------------ 辅助类与函数 ------------------
class EarlyStopping:
    """
    监控验证损失，当连续若干 epoch 无改进时提前终止训练
    """
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience  # 耐心值
        self.min_delta = min_delta  # 最小改进阈值
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def step(self, val_loss):
        # 如果是第一次调用或者损失有显著降低
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # 重置计数
        else:
            self.counter += 1
            print(f"EarlyStopping: {self.counter}/{self.patience} (val_loss not improving)")
            if self.counter >= self.patience:
                print("EarlyStopping: Stop training!")
                self.early_stop = True


def parse_args():
    """解析训练脚本的命令行参数"""
    parser = argparse.ArgumentParser(description="教师模型训练与学生蒸馏训练脚本")
    parser.add_argument('--data_dir', type=str, default='run/data', help='COCO 数据集根目录')
    parser.add_argument('--output_dir', type=str, default='run', help='输出文件目录')
    parser.add_argument('--teacher_epochs', type=int, default=2, help='教师模型训练轮数')
    parser.add_argument('--student_epochs', type=int, default=2, help='学生蒸馏训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='训练批大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='初始学习率')
    parser.add_argument('--min_delta', type=float, default=1e-4, help='EarlyStopping 最小改进阈值')
    parser.add_argument('--patience', type=int, default=5, help='EarlyStopping 耐心值（轮）')
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine'], help='学习率调度策略')
    parser.add_argument('--device', type=str,
                        default=('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'),
                        help='运行设备（cpu 或 cuda）')
    parser.add_argument('--val_viz_num', type=int, default=3,
                        help='每轮上传至 WandB 的验证样本数量')
    return parser.parse_args()


def train_one_epoch_teacher(model, loader, optimizer, seg_loss_fn, kp_loss_fn, device):
    """执行教师模型单个 epoch 的训练"""
    model.train()
    total_loss = 0.0
    for imgs, masks, kps, vs, _ in loader:
        # 将数据迁移到目标设备
        imgs, masks = imgs.to(device), masks.to(device)
        kps = [t.to(device) for t in kps]
        vs = [t.to(device) for t in vs]
        # 前向计算：得到分割 logits 与姿态 logits
        seg_logits, pose_logits = model(imgs)
        # 计算分割损失与关键点损失之和
        loss = seg_loss_fn(seg_logits, masks) + kp_loss_fn(pose_logits, kps, vs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    # 返回平均训练损失
    return total_loss / len(loader)


def evaluate_teacher(model, loader, seg_loss_fn, kp_loss_fn, device):
    """评估教师模型在验证集上的表现"""
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
    """执行学生模型单个 epoch 的蒸馏训练"""
    student.train()
    teacher.eval()  # 教师模型固定
    total_loss = 0.0
    metrics_agg = {}
    for imgs, masks, kps, vs, _ in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        kps = [t.to(device) for t in kps]
        vs = [t.to(device) for t in vs]
        with torch.no_grad():
            # 教师输出作为软标签
            t_seg, t_kp = teacher(imgs)
        # 学生前向
        s_seg, s_kp = student(imgs)
        # 计算蒸馏损失和指标
        loss, metrics = distill_loss_fn((s_seg, s_kp), (t_seg, t_kp), (masks, kps, vs))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # 累计度量值
        for k, v in metrics.items():
            metrics_agg[k] = metrics_agg.get(k, 0.0) + v
    # 平均所有度量
    metrics_avg = {k: v / len(loader) for k, v in metrics_agg.items()}
    return total_loss / len(loader), metrics_avg


def evaluate_student(student, teacher, loader, seg_loss_fn, kp_loss_fn, device):
    """评估学生模型在验证集上的表现（不使用软标签）"""
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


def log_val_images_to_wandb(loader, teacher, student, device, wandb_logger, num_images, step):
    """
    从验证集随机抽取样本，将原图、教师和学生的分割结果上传到 wandb
    """
    all_indices = list(range(len(loader.dataset)))
    sampled = random.sample(all_indices, num_images)
    log_images = []
    for idx in sampled:
        img_tensor, _, _, _, path = loader.dataset[idx]
        orig = Image.open(path)
        inp = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            t_seg, _ = teacher(inp)
            s_seg, _ = student(inp)
        # 生成为 PIL 格式的分割 mask
        t_mask = Image.fromarray((torch.sigmoid(t_seg)[0,0].cpu().numpy() * 255).astype(np.uint8)).resize(orig.size)
        s_mask = Image.fromarray((torch.sigmoid(s_seg)[0,0].cpu().numpy() * 255).astype(np.uint8)).resize(orig.size)
        # 添加到 wandb.Image 列表
        log_images.extend([
            wandb.Image(orig, caption="Original"),
            wandb.Image(t_mask, caption="Teacher"),
            wandb.Image(s_mask, caption="Student")
        ])
    # 一次性上传
    wandb_logger.log({"val_comparison": log_images}, step=step)


def infer_and_visualize(model, loader, device, outdir, prefix):
    """
    将模型在验证集上的全部预测可视化并保存到指定目录
    """
    model.eval()
    vis_dir = os.path.join(outdir, 'vis', prefix)
    os.makedirs(vis_dir, exist_ok=True)
    for imgs, masks, kps, vs, paths in loader:
        imgs = imgs.to(device)
        with torch.no_grad():
            seg_logits, kp_logits = model(imgs)
        # 将 logits 转为概率图并上采样到原始大小
        seg_prob = torch.sigmoid(seg_logits)
        seg_resized = torch.nn.functional.interpolate(seg_prob, size=masks.shape[-2:],
                                                     mode='bilinear', align_corners=False)
        seg_mask = (seg_resized > 0.5).squeeze(1).cpu().numpy()
        hm = kp_logits.cpu().numpy()
        # 构造可视化结果列表
        results = []
        for i, path in enumerate(paths):
            orig = Image.open(path)
            ow, oh = orig.size
            # 分割预测
            mask_pred = Image.fromarray((seg_mask[i] * 255).astype(np.uint8)).resize((ow, oh))
            # 关键点 GT 与预测可按 similar 方法绘制（省略）
            results.append({
                'image_path': path,
                'mask': np.array(mask_pred),
                'keypoints': [],  # 可按需要添加
                'visibility': [],
                'gt_mask': None,
                'gt_keypoints': None,
                'gt_visibility': None
            })
        # 调用外部可视化函数保存
        visualize_predictions(results, vis_dir)
    print(f"{prefix} 可视化结果已保存到 {vis_dir}")


def main():
    """主函数：组织整个训练与蒸馏流程"""
    args = parse_args()
    print(f"Using device: {args.device}")
    # 创建必要目录
    os.makedirs(args.output_dir, exist_ok=True)
    for sub in ['data', 'vis', 'models']:
        os.makedirs(os.path.join(args.output_dir, sub), exist_ok=True)

    # 加载 COCO 数据集
    train_samples = prepare_coco_dataset(args.data_dir, 'train')
    val_samples = prepare_coco_dataset(args.data_dir, 'validation')
    if not train_samples or not val_samples:
        raise RuntimeError("COCO 数据加载失败，请检查 data_dir 路径")

    # 可视化部分原始样本，便于检查数据
    visualize_raw_samples(train_samples[:5], os.path.join(args.output_dir, 'vis', 'raw'))

    # 构建 DataLoader，使用多 worker 加速
    train_loader = DataLoader(
        COCODataset(train_samples), batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        COCODataset(val_samples), batch_size=1, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True
    )

    device = torch.device(args.device)

    # ------------------ 教师模型训练 ------------------
    teacher = TeacherModel()
    # 替换或配置子模块
    teacher.segmentation = UNetSegmentation(in_channels=3, num_classes=1)
    teacher.pose = PoseEstimationModel(in_channels=4)
    teacher.to(device)
    optimizer_t = torch.optim.Adam(teacher.parameters(), lr=args.lr)
    # 选择学习率调度器
    if args.scheduler == 'plateau':
        scheduler_t = ReduceLROnPlateau(optimizer_t, mode='min', factor=0.5, patience=3)
    else:
        scheduler_t = CosineAnnealingLR(optimizer_t, T_max=args.teacher_epochs)
    earlystop_t = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
    wandb_logger_t = WandbLogger(_project="Teacher_Model",
                                 _entity="joint_angle",
                                 config=vars(args))

    # 训练循环
    for epoch in range(1, args.teacher_epochs + 1):
        train_loss = train_one_epoch_teacher(teacher, train_loader, optimizer_t,
                                             SegmentationLoss(), KeypointsLoss(), device)
        val_loss = evaluate_teacher(teacher, val_loader,
                                     SegmentationLoss(), KeypointsLoss(), device)
        print(f"[Teacher Epoch {epoch}] train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        # 更新调度器与早停
        if args.scheduler == 'plateau':
            scheduler_t.step(val_loss)
        else:
            scheduler_t.step()
        earlystop_t.step(val_loss)
        # 日志记录
        wandb_logger_t.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': optimizer_t.param_groups[0]['lr']
        }, step=epoch)
        # 上传对比图到 WandB
        log_val_images_to_wandb(val_loader, teacher, teacher, device,
                                wandb_logger_t, args.val_viz_num, epoch)
        if earlystop_t.early_stop:
            break

    # 保存教师模型
    torch.save(teacher.state_dict(), os.path.join(args.output_dir, 'models', 'teacher.pth'))
    wandb_logger_t.finish()
    print("教师模型训练完成，权重已保存。")

    # ------------------ 学生模型蒸馏训练 ------------------
    student = StudentModel(num_keypoints=17, seg_channels=1).to(device)
    optimizer_s = torch.optim.Adam(student.parameters(), lr=args.lr)
    if args.scheduler == 'plateau':
        scheduler_s = ReduceLROnPlateau(optimizer_s, mode='min', factor=0.5, patience=3)
    else:
        scheduler_s = CosineAnnealingLR(optimizer_s, T_max=args.student_epochs)
    earlystop_s = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
    wandb_logger_s = WandbLogger(_project="Student_Model",
                                 _entity="joint_angle",
                                 config=vars(args))

    # 蒸馏训练循环
    for epoch in range(1, args.student_epochs + 1):
        train_loss, metrics = train_one_epoch_student(student, teacher, train_loader,
                                                      optimizer_s, DistillationLoss(), device)
        val_loss = evaluate_student(student, teacher, val_loader,
                                     SegmentationLoss(), KeypointsLoss(), device)
        print(f"[Student Epoch {epoch}] train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        if args.scheduler == 'plateau':
            scheduler_s.step(val_loss)
        else:
            scheduler_s.step()
        earlystop_s.step(val_loss)
        log_data = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                    'lr': optimizer_s.param_groups[0]['lr']}
        log_data.update(metrics)  # 添加蒸馏相关指标
        wandb_logger_s.log(log_data, step=epoch)
        # 对比图上传（教师 vs 学生）
        log_val_images_to_wandb(val_loader, teacher, student, device,
                                wandb_logger_s, args.val_viz_num, epoch)
        if earlystop_s.early_stop:
            break

    # 保存学生模型
    torch.save(student.state_dict(), os.path.join(args.output_dir, 'models', 'student.pth'))
    wandb_logger_s.finish()
    print("学生模型蒸馏训练完成，权重已保存。")

if __name__ == '__main__':
    main()
