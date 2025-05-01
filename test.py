#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test.py: 实现完整的训练流程，包括 COCO 数据加载、可视化、教师模型训练、
学生模型蒸馏训练、推理与结果可视化。
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(__file__))

from utils.coco import prepare_coco_dataset
from utils.visualization import visualize_raw_samples, visualize_predictions
from utils.loss import SegmentationLoss, KeypointsLoss, DistillationLoss
from models.teacher_model import TeacherModel
from models.student_model import StudentModel
from models.segmentation_model import UNetSegmentation
from models.pose_model import PoseEstimationModel


def parse_args():
    parser = argparse.ArgumentParser(description="训练与验证模型")
    parser.add_argument('--data_dir', type=str, default='run/data', help='COCO 数据集目录')
    parser.add_argument('--output_dir', type=str, default='run', help='输出运行文件目录')
    parser.add_argument('--teacher_epochs', type=int, default=2, help='教师模型训练 epoch 数')
    parser.add_argument('--student_epochs', type=int, default=2, help='学生模型蒸馏训练 epoch 数')
    parser.add_argument('--batch_size', type=int, default=1, help='批大小 (推荐1避免尺寸不匹配)')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--max_samples', type=int, default=5, help='最大样本数量 (None 表示全量)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else'cpu',
                        help='设备 (cpu 或 cuda)')
    return parser.parse_args()


class COCODataset(Dataset):
    """
    PyTorch Dataset for COCO samples, with resizing to fixed dimensions for UNet compatibility.
    """

    def __init__(self, samples, img_transform=None, mask_transform=None):
        self.samples = samples
        self.img_transform = img_transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        self.mask_transform = mask_transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = Image.open(item['image_path']).convert('RGB')
        img_tensor = self.img_transform(img)

        mask_np = item['mask']
        mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))
        mask_tensor = self.mask_transform(mask_img)
        mask_tensor = (mask_tensor > 0.5).float()

        kps = item.get('keypoints', None)
        vis = item.get('visibility', None)
        if kps is None or (isinstance(kps, np.ndarray) and kps.size == 0):
            kps_tensor = torch.zeros((0, 2), dtype=torch.float32)
            vis_tensor = torch.zeros((0,), dtype=torch.int32)
        else:
            coords = kps.astype(np.float32).copy()
            coords[:, 0] = coords[:, 0] / mask_np.shape[1] * 256
            coords[:, 1] = coords[:, 1] / mask_np.shape[0] * 256
            kps_tensor = torch.from_numpy(coords)
            vis_tensor = torch.from_numpy(vis.astype(np.int32))

        return img_tensor, mask_tensor, kps_tensor, vis_tensor, item['image_path']


def collate_fn(batch):
    imgs = torch.stack([x[0] for x in batch])
    masks = torch.stack([x[1] for x in batch])
    keypoints = [x[2] for x in batch]
    visibilities = [x[3] for x in batch]
    paths = [x[4] for x in batch]
    return imgs, masks, keypoints, visibilities, paths


def train_teacher(model, loader, optimizer, seg_loss, kp_loss, device):
    model.train()
    total_loss = 0.0
    for imgs, masks, kps, vs, _ in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        kps = [t.to(device) for t in kps]
        vs = [t.to(device) for t in vs]

        seg_out, kp_out = model(imgs)
        loss = seg_loss(seg_out, masks) + kp_loss(kp_out, kps, vs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"教师模型训练平均损失: {total_loss / len(loader):.4f}")


def train_student(student, teacher, loader, optimizer, distill_loss, device):
    student.train()
    teacher.eval()
    total_loss = 0.0
    for imgs, masks, kps, vs, _ in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        kps = [t.to(device) for t in kps]
        vs = [t.to(device) for t in vs]

        with torch.no_grad():
            t_seg, t_kp = teacher(imgs)
        s_seg, s_kp = student(imgs)
        loss, _ = distill_loss((s_seg, s_kp), (t_seg, t_kp), (masks, kps, vs))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"学生蒸馏训练平均损失: {total_loss / len(loader):.4f}")


def infer_and_visualize(model, loader, device, outdir, prefix):
    model.eval()
    results = []
    vis_dir = os.path.join(outdir, 'vis', prefix)
    os.makedirs(vis_dir, exist_ok=True)
    for imgs, masks, kps, vs, paths in loader:
        imgs = imgs.to(device)
        with torch.no_grad():
            seg_out, kp_out = model(imgs)
        seg_prob = torch.sigmoid(seg_out)
        seg_resized = F.interpolate(seg_prob, size=masks.shape[-2:], mode='bilinear', align_corners=False)
        seg_bin = (seg_resized > 0.5).squeeze(1).cpu().numpy()
        hm = kp_out.cpu().numpy()

        for i, path in enumerate(paths):
            orig_img = Image.open(path)
            orig_w, orig_h = orig_img.size

            mask_pred = Image.fromarray((seg_bin[i] * 255).astype(np.uint8))
            mask_pred = mask_pred.resize((orig_w, orig_h), resample=Image.NEAREST)
            mask_pred_np = np.array(mask_pred).astype(np.uint8)

            # 生成 gt_mask
            gt_mask_img = Image.fromarray((masks[i].squeeze(0).numpy() * 255).astype(np.uint8), mode='L')
            gt_mask_resized = gt_mask_img.resize((orig_w, orig_h), resample=Image.NEAREST)
            gt_mask_np = (np.array(gt_mask_resized) > 128).astype(np.uint8)

            coords = []
            for heat in hm[i]:
                y, x = np.unravel_index(heat.argmax(), heat.shape)
                dx = x * 256 / heat.shape[1]
                dy = y * 256 / heat.shape[0]
                coord_x = dx * orig_w / 256
                coord_y = dy * orig_h / 256
                coords.append([coord_x, coord_y])

            results.append({
                'image_path': path,
                'mask': mask_pred_np,
                'keypoints': np.array(coords, dtype=np.float32),
                'visibility': np.ones((len(coords),), dtype=np.int32),
                'gt_mask': gt_mask_np,
                'gt_keypoints': (kps[i].numpy().astype(np.float32) * np.array([[orig_w/256, orig_h/256]])),
                'gt_visibility': vs[i].numpy(),
            })
    visualize_predictions(results, vis_dir)
    print(f"{prefix} 可视化保存至 {vis_dir}")


def main():
    args = parse_args()

    print(args)

    os.makedirs(args.output_dir, exist_ok=True)
    for sub in ['vis', 'models']:
        os.makedirs(os.path.join(args.output_dir, sub), exist_ok=True)
    device = torch.device(args.device)

    train_samples = prepare_coco_dataset(args.data_dir, 'train', max_samples=args.max_samples)
    if len(train_samples) == 0:
        raise ValueError(f"训练集样本数为0，请检查路径 {args.data_dir}")
    val_samples = prepare_coco_dataset(args.data_dir, 'validation', max_samples=10)

    visualize_raw_samples(train_samples[:5], os.path.join(args.output_dir, 'vis', 'raw'))

    train_loader = DataLoader(
        COCODataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        COCODataset(val_samples),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )

    teacher = TeacherModel()
    teacher.segmentation = UNetSegmentation(in_channels=3, num_classes=1)
    teacher.pose = PoseEstimationModel(in_channels=4)
    teacher.to(device)
    opt_t = torch.optim.Adam(teacher.parameters(), lr=args.lr)
    for _ in range(args.teacher_epochs):
        train_teacher(teacher, train_loader, opt_t, SegmentationLoss(), KeypointsLoss(), device)
    torch.save(teacher.state_dict(), os.path.join(args.output_dir, 'models', 'teacher.pth'))

    student = StudentModel(num_keypoints=17).to(device)
    opt_s = torch.optim.Adam(student.parameters(), lr=args.lr)
    for _ in range(args.student_epochs):
        train_student(student, teacher, train_loader, opt_s, DistillationLoss(0.5, 2.0), device)
    torch.save(student.state_dict(), os.path.join(args.output_dir, 'models', 'student.pth'))

    infer_and_visualize(teacher, val_loader, device, args.output_dir, 'teacher')
    infer_and_visualize(student, val_loader, device, args.output_dir, 'student')


if __name__ == '__main__':
    main()
