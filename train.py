# 文件：train.py

import os, time, random
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import wandb

from utils.visualization import SKELETON, COCO_KEYPOINT_NAMES
from models.SegKP_Model import SegmentKeypointModel, PosePostProcessor
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from early_stopping_pytorch import EarlyStopping

from utils.coco import prepare_coco_dataset, COCODataset, collate_fn
from utils.loss import criterion
from utils.Evaluator import SegmentationEvaluator, PoseEvaluator

from config import arg_real, arg_test

# 可视化辅助函数
def create_mask_visual(mask):
    vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
    vis[mask > 0] = [255, 0, 0]
    return vis

def create_pose_visual(kps_list, image_size, skeleton):
    h, w = image_size
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    for person_kps in kps_list:
        # 绘制骨骼线
        for (a, b) in skeleton:
            if a < len(person_kps) and b < len(person_kps):
                x1, y1 = person_kps[a]
                x2, y2 = person_kps[b]
                if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                    cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        # 绘制关键点
        for (x, y) in person_kps:
            if x > 0 and y > 0:
                cv2.circle(vis, (int(x), int(y)), 3, (0,0,255), -1)
    return vis

def log_samples(seg_gts, seg_preds, pose_gts, pose_preds, original_sizes, epoch):
    """将几个验证样本的GT与预测结果逐张图片上传到WandB，并带上命名"""
    post_proc = PosePostProcessor()
    inds = random.sample(range(len(seg_gts)), min(len(seg_gts), 2))

    log_dict = {}
    for i, idx in enumerate(inds):
        orig_h, orig_w = original_sizes[idx]

        # GT 分割可视化
        gt_mask = seg_gts[idx].cpu().numpy().squeeze().astype(np.uint8)
        gt_vis = create_mask_visual(gt_mask)
        # 预测分割可视化
        pred_logit = seg_preds[idx].cpu().detach().numpy().squeeze()
        pred_mask = (torch.sigmoid(torch.from_numpy(pred_logit).unsqueeze(0)) > 0.5).numpy()[0]
        pred_vis = create_mask_visual(pred_mask.astype(np.uint8))

        # GT 关键点可视化
        gt_hm = pose_gts[idx]
        gt_hm = torch.nn.functional.interpolate(
            gt_hm.unsqueeze(0), size=(orig_h, orig_w),
            mode='bilinear', align_corners=False
        )[0]
        gt_kps = post_proc(gt_hm.unsqueeze(0))[0]
        gt_pose = create_pose_visual(gt_kps, (orig_h, orig_w), SKELETON)

        # 预测关键点可视化
        pred_hm = pose_preds[idx]
        pred_hm = torch.nn.functional.interpolate(
            pred_hm.unsqueeze(0), size=(orig_h, orig_w),
            mode='bilinear', align_corners=False
        )[0]
        pred_kps = post_proc(pred_hm.unsqueeze(0))[0]
        pred_pose = create_pose_visual(pred_kps, (orig_h, orig_w), SKELETON)

        # 按命名收集到日志字典
        log_dict[f"val/gt_seg_epoch{epoch}_sample{i}"]  = wandb.Image(gt_vis,   caption=f"GT Seg {idx}")
        log_dict[f"val/pred_seg_epoch{epoch}_sample{i}"] = wandb.Image(pred_vis, caption=f"Pred Seg {idx}")
        log_dict[f"val/gt_pose_epoch{epoch}_sample{i}"] = wandb.Image(gt_pose,  caption=f"GT Pose {idx}")
        log_dict[f"val/pred_pose_epoch{epoch}_sample{i}"] = wandb.Image(pred_pose,caption=f"Pred Pose {idx}")

    # 一次性上传所有图片
    wandb.log(log_dict, step=epoch)

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 数据加载
    train_samples = prepare_coco_dataset(args.data_dir, split='train', max_samples=args.max_samples)
    val_samples   = prepare_coco_dataset(args.data_dir, split='val',   max_samples=args.max_samples)
    train_loader = DataLoader(COCODataset(train_samples), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    val_loader   = DataLoader(COCODataset(val_samples),   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    # 模型、优化器、学习率调度器、混合精度、EarlyStopping
    model      = SegmentKeypointModel().to(device)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler  = CosineAnnealingWarmupRestarts(
        optimizer, first_cycle_steps=args.epochs*len(train_loader),
        cycle_mult=1.0, max_lr=args.lr, min_lr=1e-6,
        warmup_steps=args.warmup_epochs*len(train_loader), gamma=1.0)
    scaler     = GradScaler(enabled=args.use_fp16)
    early_stop = EarlyStopping(patience=args.patience, delta=args.min_delta)
    post_proc  = PosePostProcessor()

    # WandB 初始化
    wandb.init(project=args.project_name, config=vars(args), entity=args.entity)
    best_ap = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        epoch_train_start = time.time()

        pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for batch_idx, (imgs, masks, hm, paf, _, _, _) in enumerate(train_loader, start=1):
            imgs, masks, hm, paf = imgs.to(device), masks.to(device), hm.to(device), paf.to(device)

            optimizer.zero_grad()
            if args.use_fp16:
                with autocast(device_type="cuda"):
                    seg_pred, pose_pred = model(imgs)  # 训练时返回 (seg_logits, pose_heatmaps)
                    loss = criterion(seg_pred, masks, pose_pred, hm, paf)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                seg_pred, pose_pred = model(imgs)
                loss = criterion(seg_pred, masks, pose_pred, hm, paf)
                loss.backward()
                optimizer.step()

            scheduler.step()
            train_loss += loss.item()
            # Batch 日志
            lr = optimizer.param_groups[0]['lr']

            total_norm_sqr = torch.zeros(1, device=device)
            for p in model.parameters():
                if p.grad is not None:
                    total_norm_sqr += p.grad.norm(2) ** 2
            grad_norm = torch.sqrt(total_norm_sqr).item()

            wandb.log({
                'train/loss': loss.item(),
                'train/lr':   lr,
                'train/grad_norm': grad_norm
            }, step=(epoch-1)*len(train_loader) + batch_idx)

            pbar.update(1)
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{lr:.2e}"})
        pbar.close()
        avg_train_loss = train_loss / len(train_loader)
        train_time = time.time() - epoch_train_start

        # 验证阶段
        model.eval()
        val_loss = 0.0
        epoch_val_start = time.time()
        all_pred_masks, all_gt_masks = [], []
        sample_seg_gts, sample_seg_preds = [], []
        sample_pose_gts, sample_pose_preds = [], []
        orig_sizes = []
        with torch.no_grad():
            for (imgs, masks, hm, paf, _, _, paths) in val_loader:
                imgs, masks, hm, paf = imgs.to(device), masks.to(device), hm.to(device), paf.to(device)
                seg_pred, pose_pred = model(imgs)  # 推理时 model 返回 (seg_logits, pose_heatmaps)
                loss = criterion(seg_pred, masks, pose_pred, hm, paf)
                val_loss += loss.item()

                # 二值化分割 mask 并保存至列表
                pm = (torch.sigmoid(seg_pred) > 0.5).cpu().numpy().astype(np.uint8)
                for b in range(pm.shape[0]):
                    h_i, w_i = masks.shape[2], masks.shape[3]
                    m_uint8 = pm[b,0]
                    m_res = cv2.resize(m_uint8, (w_i, h_i), interpolation=cv2.INTER_NEAREST)
                    all_pred_masks.append(m_res)
                    all_gt_masks.append(masks[b,0].cpu().numpy())

                    sample_seg_gts.append(masks[b])
                    sample_seg_preds.append(seg_pred[b])
                    sample_pose_gts.append(hm[b])
                    sample_pose_preds.append(pose_pred[b])
                    orig_sizes.append((h_i, w_i))

        avg_val_loss = val_loss / len(val_loader)
        val_time = time.time() - epoch_val_start

        # 计算分割 mIoU
        seg_eval = SegmentationEvaluator(num_classes=2)
        for pm_arr, gt_arr in zip(all_pred_masks, all_gt_masks):
            seg_eval.update(torch.from_numpy(pm_arr[None,None,:,:]), torch.from_numpy(gt_arr[None,None,:,:]))
        seg_iou = seg_eval.compute_miou()

        # 计算姿态 AP
        pose_eval = PoseEvaluator(os.path.join(args.data_dir, "annotations", "person_keypoints_val2017.json"),
                                  COCO_KEYPOINT_NAMES)
        batch_preds = []
        for img_id, hm_arr in enumerate(sample_pose_preds):
            # hm_arr: 17xHxW 热图
            persons = post_proc(hm_arr.unsqueeze(0))[0]  # 提取关键点
            scores = [1.0] * len(persons)
            batch_preds.append((img_id, persons, scores))
        pose_eval.update(batch_preds, None)
        pose_ap = pose_eval.compute_ap()

        # 如果 Pose AP 提升，则保存模型 checkpoint
        if pose_ap > best_ap:
            best_ap = pose_ap
            os.makedirs(os.path.join(args.output_dir, "models"), exist_ok=True)
            best_path = os.path.join(args.output_dir, "models", f"best_keypoint_model_epoch{epoch}.pth")
            torch.save(model.state_dict(), best_path)

            artifact = wandb.Artifact(
                name=f"best-keypoint-model", type="model",
                description=f"Epoch {epoch}, pose_ap={best_ap:.4f}"
            )
            artifact.add_file(best_path)
            artifact.metadata = {"epoch": epoch, "seg_iou": seg_iou, "pose_ap": pose_ap}
            wandb.log_artifact(artifact, aliases=["best-keypoint"])
            wandb.run.summary["best_pose_ap"] = best_ap

        # 记录 epoch 级日志
        wandb.log({
            'train/avg_loss': avg_train_loss,
            'val/avg_loss':   avg_val_loss,
            'train/epoch_time': train_time,
            'val/seg_iou':    seg_iou,
            'val/pose_ap':    pose_ap,
            'val/epoch_time': val_time,
        }, step=epoch)

        # 上传可视化样本到 WandB
        log_samples(sample_seg_gts, sample_seg_preds,
                    sample_pose_gts, sample_pose_preds,
                    orig_sizes, epoch)

        # EarlyStopping 判断
        if early_stop(avg_val_loss, model):
            print("Early stopping triggered")
            break

    # 保存最终模型
    os.makedirs(os.path.join(args.output_dir, "models"), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "models", "model_final.pth"))

# 主函数入口：使用 arg_real() 或 arg_test() 解析参数
if __name__ == "__main__":
    args = arg_test()  # 或根据需求使用 arg_test()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
    torch.distributed.destroy_process_group()
