import os
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pycocotools.cocoeval import COCOeval
import wandb
import cv2
from tqdm import tqdm
from scipy.ndimage import maximum_filter
from PIL import Image

from models.Multi_Pose import MultiPoseNet

from utils.visualization import visualize_coco_keypoints
from utils.loss import PoseLoss
from utils.coco import COCOPoseDataset, COCO_PERSON_SKELETON


NUM_KP=17
# -----------------------
# Evaluation and Visualization
# -----------------------
def evaluate(model, val_loader, device):
    """
    Evaluate model on COCO val set; return mean AP, AP50, and a list of WandB Images.
    Includes sub-pixel refinement, PAF-based grouping, and custom visualization.
    """
    model.eval()
    coco_gt = val_loader.dataset.coco

    results = []
    img_ids = []
    vis_list = []

    # 随机选取 n_vis 张图做可视化
    n_vis = getattr(wandb.config, 'n_vis', 3)
    vis_ids = random.sample(val_loader.dataset.img_ids, min(n_vis, len(val_loader.dataset.img_ids)))
    idx_offset = 0

    with torch.no_grad():
        for imgs, _, _ in val_loader:
            imgs = imgs.to(device)
            output = model(imgs)
            # 兼容可能的多输出（refine 或单阶段）
            if isinstance(output, (tuple, list)):
                if len(output) == 4:
                    heat_pred, paf_pred = output[0], output[1]
                elif len(output) == 2:
                    heat_pred, paf_pred = output
                else:
                    raise ValueError(f"evaluate: unexpected model output length {len(output)}")
            else:
                heat_pred, paf_pred = output, None

            B, H, W = heat_pred.shape[0], heat_pred.shape[2], heat_pred.shape[3]

            for i in range(B):
                img_id = val_loader.dataset.img_ids[idx_offset + i]
                img_ids.append(img_id)

                heat = heat_pred[i].cpu().numpy()       # shape: [K, H, W]
                paf = paf_pred[i].cpu().numpy() if paf_pred is not None else None

                # 1. 各关键点通道的峰值提取 + 亚像素精细化
                all_peaks = []
                peak_thresh = 0.1
                for k in range(heat.shape[0]):
                    hmap = heat[k]
                    # 局部极大 + 阈值
                    max_f = maximum_filter(hmap, size=3)
                    mask = (hmap == max_f) & (hmap > peak_thresh)
                    coords = np.argwhere(mask)  # [[y, x], ...]
                    refined_peaks = []
                    for (y, x) in coords:
                        score = float(hmap[y, x])
                        # Taylor expansion for subpixel 修正
                        dx = dy = 0.0
                        if 0 < x < W - 1 and 0 < y < H - 1:
                            dx = 0.5 * (hmap[y, x+1] - hmap[y, x-1])
                            dy = 0.5 * (hmap[y+1, x] - hmap[y-1, x])
                            dxx = hmap[y, x+1] + hmap[y, x-1] - 2*hmap[y, x]
                            dyy = hmap[y+1, x] + hmap[y-1, x] - 2*hmap[y, x]
                            if dxx != 0: dx /= -dxx
                            if dyy != 0: dy /= -dyy
                        refined_peaks.append((float(x) + dx, float(y) + dy, score))
                    all_peaks.append(refined_peaks)

                # 2. 构建 PAF 连接候选，并贪心匹配
                connection_candidates = {c: [] for c in range(len(COCO_PERSON_SKELETON))}
                if paf is not None:
                    for c, (a, b) in enumerate(COCO_PERSON_SKELETON):
                        candA = all_peaks[a]
                        candB = all_peaks[b]
                        if not candA or not candB:
                            continue
                        paf_x = paf[2*c]
                        paf_y = paf[2*c+1]
                        for ia, (ax, ay, sa) in enumerate(candA):
                            for ib, (bx, by, sb) in enumerate(candB):
                                dx, dy = bx - ax, by - ay
                                norm = np.hypot(dx, dy) + 1e-8
                                vx, vy = dx / norm, dy / norm
                                # 沿连线均匀采样
                                num_samples = 10
                                xs = np.linspace(ax, bx, num_samples).astype(int)
                                ys = np.linspace(ay, by, num_samples).astype(int)
                                scores = []
                                for xx, yy in zip(xs, ys):
                                    if 0 <= xx < W and 0 <= yy < H:
                                        vec = np.array([paf_x[yy, xx], paf_y[yy, xx]])
                                        scores.append(vec.dot([vx, vy]))
                                if not scores:
                                    continue
                                avg_score = float(np.mean(scores))
                                if avg_score > 0 and np.mean((np.array(scores) > 0.05).astype(float)) > 0.8:
                                    total_score = avg_score + 0.5 * (sa + sb)
                                    connection_candidates[c].append((ia, ib, total_score))
                        # 贪心选取不冲突连接
                        connection_candidates[c].sort(key=lambda x: x[2], reverse=True)
                        usedA, usedB = set(), set()
                        conns = []
                        for ia, ib, _ in connection_candidates[c]:
                            if ia not in usedA and ib not in usedB:
                                conns.append((ia, ib))
                                usedA.add(ia); usedB.add(ib)
                        connection_candidates[c] = conns

                # 3. 组装多人骨架实例
                persons = []
                for c, (a, b) in enumerate(COCO_PERSON_SKELETON):
                    for ia, ib in connection_candidates.get(c, []):
                        placed = False
                        for p in persons:
                            if (a in p and p[a] == ia) or (b in p and p[b] == ib):
                                p[a] = ia; p[b] = ib; placed = True; break
                        if not placed:
                            persons.append({a: ia, b: ib})
                # 孤立关键点也作为单实例
                for k, peaks in enumerate(all_peaks):
                    for idx_p in range(len(peaks)):
                        if not any(p.get(k) == idx_p for p in persons):
                            persons.append({k: idx_p})

                # 4. 转为 COCO 评估格式
                orig = coco_gt.loadImgs([img_id])[0]
                for p in persons:
                    kps_flat = []
                    scores = []
                    for k in range(len(all_peaks)):
                        if k in p:
                            x, y, sc = all_peaks[k][p[k]]
                            x *= (orig['width'] / W)
                            y *= (orig['height'] / H)
                        else:
                            x, y, sc = 0.0, 0.0, 0.0
                        kps_flat.extend([float(x), float(y), float(sc)])
                        scores.append(sc)
                    results.append({
                        'image_id': img_id,
                        'category_id': 1,
                        'keypoints': kps_flat,
                        'score': float(np.mean(scores))
                    })

                # 5. 可视化 GT(green) vs Pred(red)
                if img_id in vis_ids:
                    img_info = coco_gt.loadImgs([img_id])[0]
                    img_path = os.path.join(
                        val_loader.dataset.root,
                        val_loader.dataset.img_folder,
                        img_info['file_name']
                    )
                    orig_img = cv2.imread(img_path)
                    if orig_img is None:
                        orig_img = np.zeros((img_info['height'], img_info['width'], 3), dtype=np.uint8)
                    tw, th = val_loader.dataset.img_size[1], val_loader.dataset.img_size[0]

                    # 先画 GT
                    gt_anns = coco_gt.loadAnns(
                        coco_gt.getAnnIds(imgIds=[img_id], catIds=[1], iscrowd=None)
                    )
                    vis_img = visualize_coco_keypoints(orig_img, gt_anns, COCO_PERSON_SKELETON,(tw, th),(0, 255, 0),(0, 255, 0))

                    # 再画 Pred
                    pred_anns = []
                    for p in persons:
                        kplist = []
                        num_kp = 0
                        for k in range(len(all_peaks)):
                            if k in p:
                                x, y, sc = all_peaks[k][p[k]]
                                x *= (orig['width'] / W)
                                y *= (orig['height'] / H)
                                v = 2 if sc > 0 else 0
                                num_kp += 1
                            else:
                                x, y, v = 0.0, 0.0, 0
                            kplist.extend([x, y, v])
                        pred_anns.append({'keypoints': kplist, 'num_keypoints': num_kp})

                    vis_img = visualize_coco_keypoints(vis_img, pred_anns, COCO_PERSON_SKELETON,(tw, th),(0, 0, 255), (0, 0, 255))

                    rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)  # numpy array, H×W×3, uint8
                    vis_list.append(
                        wandb.Image(
                            rgb,
                            caption=f"Image {img_id} – GT(green) vs Pred(red)"
                        )
                    )

            idx_offset += B

    # 6. 运行 COCOeval 并返回指标
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='keypoints')
    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mean_ap = coco_eval.stats[0]   # OKS=0.50:0.95
    ap50    = coco_eval.stats[1]   # OKS=0.50

    return mean_ap, ap50, vis_list
# -----------------------
# Training Loop
# -----------------------
def train_one_epoch(model, loader, criterion, optimizer, device, teacher=None, distill_weight=0.0):
    model.train()
    if teacher is not None:
        teacher.eval()
    epoch_loss = 0.0

    for batch_idx, (imgs, heatmaps, pafs) in enumerate(tqdm(loader, desc="Training:", leave=False)):
        imgs = imgs.to(device)
        heatmaps = heatmaps.to(device)
        pafs = pafs.to(device)

        optimizer.zero_grad()
        # 前向传播
        student_out = model(imgs)
        # 监督损失
        loss = criterion(student_out, (heatmaps, pafs))

        # 知识蒸馏（如有教师模型）
        if teacher is not None:
            with torch.no_grad():
                teacher_out = teacher(imgs)
            # 提取教师输出
            if isinstance(teacher_out, (tuple, list)):
                teacher_heat = teacher_out[0]
                teacher_paf  = teacher_out[1] if len(teacher_out) > 1 else None
            else:
                teacher_heat, teacher_paf = teacher_out, None
            # 提取学生输出
            if isinstance(student_out, (tuple, list)):
                student_heat = student_out[0]
                student_paf  = student_out[1] if len(student_out) > 1 else None
            else:
                student_heat, student_paf = student_out, None
            # 计算蒸馏MSE损失
            distill_loss = 0.0
            if teacher_heat is not None:
                distill_loss += F.mse_loss(student_heat, teacher_heat)
            if teacher_paf is not None and student_paf is not None:
                distill_loss += F.mse_loss(student_paf, teacher_paf)
            loss = loss + distill_weight * distill_loss

        loss.backward()
        # 计算梯度范数
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        optimizer.step()
        epoch_loss += loss.item() * imgs.size(0)

    return epoch_loss / len(loader.dataset)

# -----------------------
# Main
# -----------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="轻量化多人姿态估计训练脚本")
    parser.add_argument('--data_root',   type=str,   default='run/data', help='COCO 数据集根目录')
    parser.add_argument('--batch_size',  type=int,   default=32,         help='训练批大小')
    parser.add_argument('--lr',          type=float, default=1e-3,       help='初始学习率')
    parser.add_argument('--epochs',      type=int,   default=50,         help='训练轮数')
    parser.add_argument('--img_h',       type=int,   default=256,        help='输入图像高度')
    parser.add_argument('--img_w',       type=int,   default=192,        help='输入图像宽度')
    parser.add_argument('--hm_h',        type=int,   default=64,         help='输出热图高度')
    parser.add_argument('--hm_w',        type=int,   default=48,         help='输出热图宽度')
    parser.add_argument('--sigma',       type=int,   default=2,          help='高斯热图 sigma')
    parser.add_argument('--ohkm_k',      type=int,   default=8,          help='OHKM 困难关键点 topK')
    parser.add_argument('--num_workers', type=int,   default=8,          help='DataLoader 线程数')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模型保存目录
    save_dir = 'run/models'
    os.makedirs(save_dir, exist_ok=True)

    # 初始化学生模型、损失、优化器、调度器
    model = MultiPoseNet(num_keypoints=NUM_KP, width_mult=1.0, refine=True).to(device)
    criterion = PoseLoss(ohkm_k=args.ohkm_k)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # 数据集与 DataLoader
    train_ds = COCOPoseDataset(
        root=args.data_root,
        ann_file=os.path.join(args.data_root, 'annotations/person_keypoints_train2017.json'),
        img_folder='train2017',
        img_size=(args.img_h, args.img_w),
        hm_size=(args.hm_h, args.hm_w),
        sigma=args.sigma,
        augment=True
    )
    val_ds = COCOPoseDataset(
        root=args.data_root,
        ann_file=os.path.join(args.data_root, 'annotations/person_keypoints_val2017.json'),
        img_folder='val2017',
        img_size=(args.img_h, args.img_w),
        hm_size=(args.hm_h, args.hm_w),
        sigma=args.sigma,
        augment=False
    )
    print()

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"-----------CONFIG--------------")
    for name, value in vars(args).items():
        print(f"  {name:15s} = {value}")

    print(f"  Train samples:\t{len(train_ds)}")
    print(f"  Val samples:\t{len(val_ds)}")
    print(f"-------------------------------")

    wandb.init(project='Multi_Pose_test', entity="joint_angle",config=vars(args))

    best_ap = 0.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        # 训练
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        scheduler.step()
        wandb.log({
            "train/epoch_loss": train_loss,
            "train/lr": scheduler.get_last_lr()[0],
            "epoch": epoch
        })

        # 验证
        mean_ap, ap50, vis_images = evaluate(model, val_loader, device)

        wandb.log({
                "val/mAP": mean_ap,
                "val/AP50": ap50,
                "val/examples": vis_images
        })

        elapsed = time.time() - start
        print(f"[Epoch {epoch}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f}  "
              f"mAP: {mean_ap:.4f}  AP50: {ap50:.4f}  "
              f"Time: {elapsed:.1f}s")

        # 保存最佳模型
        if mean_ap > best_ap:
            best_ap = mean_ap
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))

