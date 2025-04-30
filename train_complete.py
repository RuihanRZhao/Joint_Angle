#!/usr/bin/env python
# coding: utf-8

import os
import time
import argparse
import numpy as np
from datetime import datetime
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.segmentation_model import UNetSegmentation
from models.pose_model import PoseEstimationModel
from models.teacher_model import TeacherModel
from models.student_model import StudentModel

from utils.loss import SegmentationLoss, KeypointsLoss, DistillationLoss

from datasets.coco import CocoSegKeypoints, get_transform, collate_fn

# ---------------------- 参数解析 ----------------------
parser = argparse.ArgumentParser(description='人体分割和姿态估计训练脚本')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs_seg', type=int, default=50)
parser.add_argument('--epochs_pose', type=int, default=50)
parser.add_argument('--epochs_distill', type=int, default=100)
parser.add_argument('--lr_seg', type=float, default=1e-4)
parser.add_argument('--lr_pose', type=float, default=1e-4)
parser.add_argument('--lr_distill', type=float, default=1e-4)
parser.add_argument('--data_path', type=str, default='/path/to/coco')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
parser.add_argument('--log_dir', type=str, default='logs')
parser.add_argument('--num_keypoints', type=int, default=17)
parser.add_argument('--seg_channels', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--wandb_project', type=str, default='human-pose-seg')
parser.add_argument('--wandb_entity', type=str, default=None)
args = parser.parse_args()

os.makedirs(args.checkpoint_dir, exist_ok=True)
os.makedirs(args.log_dir, exist_ok=True)
device = torch.device(args.device)

# ---------------------- WandB 初始化 ----------------------
wandb.init(
    project=args.wandb_project,
    entity=args.wandb_entity,
    config=vars(args),
    dir=args.log_dir
)


# ---------------------- 数据加载函数（需自定义） ----------------------



def create_dataloaders(args):
    """创建数据加载器"""
    # 训练集
    train_dataset = CocoSegKeypoints(
        root=os.path.join(args.data_root, 'train2017'),
        annFile=os.path.join(args.data_root, 'annotations/person_keypoints_train2017.json'),
        transform=get_transform(train=True)
    )

    # 验证集
    val_dataset = CocoSegKeypoints(
        root=os.path.join(args.data_root, 'val2017'),
        annFile=os.path.join(args.data_root, 'annotations/person_keypoints_val2017.json'),
        transform=get_transform(train=False)
    )

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader


# ---------------------- 分割训练 ----------------------
def train_segmentation(model, train_loader, val_loader, device, args):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr_seg)
    criterion = SegmentationLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    best_val_loss = float('inf')
    wandb.watch(model, criterion, log="all", log_freq=20)

    for epoch in range(args.epochs_seg):
        model.train()
        train_loss = 0
        for batch_idx, (imgs, masks, _, _) in enumerate(train_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, masks, _, _ in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        wandb.log({'seg/train_loss': train_loss, 'seg/val_loss': val_loss, 'seg/lr': optimizer.param_groups[0]['lr']}, step=epoch)
        print(f"[Seg] Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'best_segmentation.pth'))
            wandb.save(os.path.join(args.checkpoint_dir, 'best_segmentation.pth'))

    model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'best_segmentation.pth')))
    return model

# ---------------------- 姿态训练 ----------------------
def train_pose(model, seg_model, train_loader, val_loader, device, args):
    model.to(device)
    seg_model.to(device)
    seg_model.eval()
    optimizer = optim.Adam(model.parameters(), lr=args.lr_pose)
    criterion = KeypointsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    best_val_loss = float('inf')
    wandb.watch(model, criterion, log="all", log_freq=20)

    for epoch in range(args.epochs_pose):
        model.train()
        train_loss = 0
        for batch_idx, (imgs, masks, keypoints, visibilities) in enumerate(train_loader):
            imgs = imgs.to(device)
            keypoints = keypoints.to(device)
            visibilities = visibilities.to(device)
            with torch.no_grad():
                seg_logits = seg_model(imgs)
            combined_input = torch.cat([imgs, seg_logits], dim=1)
            optimizer.zero_grad()
            pose_logits = model(combined_input)
            loss = criterion(pose_logits, keypoints, visibilities)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, masks, keypoints, visibilities in val_loader:
                imgs = imgs.to(device)
                keypoints = keypoints.to(device)
                visibilities = visibilities.to(device)
                seg_logits = seg_model(imgs)
                combined_input = torch.cat([imgs, seg_logits], dim=1)
                pose_logits = model(combined_input)
                loss = criterion(pose_logits, keypoints, visibilities)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        wandb.log({'pose/train_loss': train_loss, 'pose/val_loss': val_loss, 'pose/lr': optimizer.param_groups[0]['lr']}, step=epoch)
        print(f"[Pose] Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'best_pose.pth'))
            wandb.save(os.path.join(args.checkpoint_dir, 'best_pose.pth'))

    model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'best_pose.pth')))
    return model

# ---------------------- 蒸馏训练 ----------------------
def train_distillation(student_model, teacher_model, train_loader, val_loader, device, args):
    student_model.to(device)
    teacher_model.to(device)
    teacher_model.eval()
    optimizer = optim.Adam(student_model.parameters(), lr=args.lr_distill)
    criterion = DistillationLoss(alpha=0.5, temperature=2.0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    best_val_loss = float('inf')
    wandb.watch(student_model, log="all", log_freq=20)

    for epoch in range(args.epochs_distill):
        student_model.train()
        train_loss = 0
        for batch_idx, (imgs, masks, keypoints, visibilities) in enumerate(train_loader):
            imgs = imgs.to(device)
            masks = masks.to(device)
            keypoints = keypoints.to(device)
            visibilities = visibilities.to(device)
            with torch.no_grad():
                teacher_seg_logits, teacher_pose_logits = teacher_model(imgs)
            optimizer.zero_grad()
            student_seg_logits, student_pose_logits = student_model(imgs)
            loss, metrics = criterion(
                (student_seg_logits, student_pose_logits),
                (teacher_seg_logits, teacher_pose_logits),
                (masks, keypoints, visibilities)
            )
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        student_model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, masks, keypoints, visibilities in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                keypoints = keypoints.to(device)
                visibilities = visibilities.to(device)
                teacher_seg_logits, teacher_pose_logits = teacher_model(imgs)
                student_seg_logits, student_pose_logits = student_model(imgs)
                loss, metrics = criterion(
                    (student_seg_logits, student_pose_logits),
                    (teacher_seg_logits, teacher_pose_logits),
                    (masks, keypoints, visibilities)
                )
                val_loss += loss.item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        wandb.log({'distill/train_loss': train_loss, 'distill/val_loss': val_loss, 'distill/lr': optimizer.param_groups[0]['lr']}, step=epoch)
        print(f"[Distill] Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(student_model.state_dict(), os.path.join(args.checkpoint_dir, 'best_student.pth'))
            wandb.save(os.path.join(args.checkpoint_dir, 'best_student.pth'))

    student_model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'best_student.pth')))
    return student_model

# ---------------------- 主流程 ----------------------
def main():
    train_loader, val_loader = create_dataloaders(args)

    seg_model = UNetSegmentation(in_channels=3, num_classes=args.seg_channels)
    pose_model = PoseEstimationModel(in_channels=3 + args.seg_channels, num_keypoints=args.num_keypoints)

    print("===== 阶段1：训练分割网络 =====")
    seg_model = train_segmentation(seg_model, train_loader, val_loader, device, args)

    print("===== 阶段2：训练姿态估计网络 =====")
    pose_model = train_pose(pose_model, seg_model, train_loader, val_loader, device, args)

    teacher_model = TeacherModel(seg_model, pose_model)
    student_model = StudentModel(num_keypoints=args.num_keypoints, seg_channels=args.seg_channels)

    print("===== 阶段3：知识蒸馏 =====")
    student_model = train_distillation(student_model, teacher_model, train_loader, val_loader, device, args)

    # 保存最终模型
    torch.save(teacher_model.state_dict(), os.path.join(args.checkpoint_dir, 'final_teacher.pth'))
    torch.save(student_model.state_dict(), os.path.join(args.checkpoint_dir, 'final_student.pth'))
    wandb.save(os.path.join(args.checkpoint_dir, 'final_teacher.pth'))
    wandb.save(os.path.join(args.checkpoint_dir, 'final_student.pth'))
    print("训练完成，模型已保存。")

    # 上传整个checkpoint目录
    artifact = wandb.Artifact("all_checkpoints", type="model")
    artifact.add_dir(args.checkpoint_dir)
    wandb.log_artifact(artifact)

    wandb.finish()

if __name__ == "__main__":
    main()
