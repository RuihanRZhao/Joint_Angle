import os
import time
import cv2
import random
import numpy as np
import torch
from torch.distributed import destroy_process_group
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

from config import arg_test, arg_real


def create_mask_visual(mask):
    vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
    vis[mask > 0] = [255, 0, 0]
    return vis


def create_pose_visual(kps_list, image_size, skeleton):
    h, w = image_size
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    for person_kps in kps_list:
        for (a, b) in skeleton:
            if a < len(person_kps) and b < len(person_kps):
                x1, y1 = person_kps[a]
                x2, y2 = person_kps[b]
                if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                    cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0),2)
        for (x,y) in person_kps:
            if x>0 and y>0:
                cv2.circle(vis, (int(x),int(y)), 3, (0,0,255), -1)
    return vis


def log_samples(seg_gts, seg_preds, pose_gts, pose_preds, original_sizes, epoch, skeleton=SKELETON):
    table = wandb.Table(columns=["GT Segmentation","Pred Segmentation","GT Keypoints","Pred Keypoints"])
    post_proc = PosePostProcessor()
    inds = random.sample(range(len(seg_gts)), min(2,len(seg_gts)))
    for idx in inds:
        h, w = original_sizes[idx]
        # Seg
        gt_mask = seg_gts[idx].cpu().numpy().squeeze()
        gt_mask = cv2.resize(gt_mask, (w,h), interpolation=cv2.INTER_NEAREST)
        gt_vis = create_mask_visual(gt_mask)
        pred_logit = seg_preds[idx].cpu().numpy().squeeze()
        pred_mask = cv2.resize(pred_logit, (w,h), interpolation=cv2.INTER_LINEAR)
        pred_mask = (torch.sigmoid(torch.from_numpy(pred_mask))>0.5).numpy()
        pred_vis = create_mask_visual(pred_mask)
        # Pose
        gt_hm = pose_gts[idx].cpu().numpy()
        gt_hm = torch.nn.functional.interpolate(torch.from_numpy(gt_hm).unsqueeze(0),
                                                size=(h,w), mode='bilinear', align_corners=False)[0].numpy()
        gt_kps = post_proc(gt_hm[None])[0]
        gt_pose = create_pose_visual(gt_kps,(h,w),skeleton)
        pred_hm = pose_preds[idx].cpu().numpy()
        pred_hm = torch.nn.functional.interpolate(torch.from_numpy(pred_hm).unsqueeze(0),
                                                  size=(h,w), mode='bilinear', align_corners=False)[0].numpy()
        pred_kps = post_proc(pred_hm[None])[0]
        pred_pose = create_pose_visual(pred_kps,(h,w),skeleton)
        table.add_data(wandb.Image(gt_vis), wandb.Image(pred_vis),
                       wandb.Image(gt_pose), wandb.Image(pred_pose))
    wandb.log({f"Validation Samples Epoch {epoch}": table}, step=epoch)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # COCO 骨骼关键点注释文件路径
    ann_file = os.path.join(args.data_dir, "annotations", "person_keypoints_val2017.json")

    # 数据加载
    train_s = prepare_coco_dataset(args.data_dir, split='train', max_samples=args.max_samples)
    val_s   = prepare_coco_dataset(args.data_dir, split='val',   max_samples=args.max_samples)
    train_loader = DataLoader(
        COCODataset(train_s), batch_size=args.batch_size,
        shuffle=True,  num_workers=args.num_workers,
        pin_memory=True, collate_fn=collate_fn
    )
    val_loader   = DataLoader(
        COCODataset(val_s),   batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True, collate_fn=collate_fn
    )

    # 模型、优化器、调度器、AMP、EarlyStopping
    model      = SegmentKeypointModel().to(device)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler  = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=args.epochs * len(train_loader),
        cycle_mult=1.0,
        max_lr=args.lr,
        min_lr=1e-6,
        warmup_steps=args.warmup_epochs * len(train_loader),
        gamma=1.0
    )
    scaler     = GradScaler(enabled=args.use_fp16)
    early_stop = EarlyStopping(patience=args.patience, delta=args.min_delta)
    post_proc  = PosePostProcessor()

    # WandB 初始化
    wandb.init(project=args.project_name, config=vars(args), entity=args.entity)

    best_ap = -float('inf')

    for epoch in range(1, args.epochs + 1):
        # —— 训练阶段 —— #
        # …（训练部分保持不变）…

        # —— 验证 & 评估阶段 —— #
        # …（前面验证采集部分保持不变）…

        # 分割 mIoU 评估
        seg_eval = SegmentationEvaluator(num_classes=2)
        for pm, gt in zip(all_pred_masks, all_gt_masks):
            pm_t = torch.from_numpy(pm).unsqueeze(0).unsqueeze(0)
            gt_t = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0)
            seg_eval.update(pm_t, gt_t)
        seg_iou = seg_eval.compute_miou()

        # 关键点 AP 评估
        pose_eval = PoseEvaluator(ann_file, COCO_KEYPOINT_NAMES)
        batch_preds = []
        for img_id, hm_arr in enumerate(all_heatmaps):
            # hm_arr: numpy array shape (C, H, W)
            # 使用 numpy.ndarray.max，保持为 ndarray，使其支持 .mean()
            scores = hm_arr.max(axis=(1,2))  # numpy array of shape (C,)
            batch_preds.append((img_id, hm_arr, scores))
        pose_eval.update(batch_preds, None)
        kp_acc = pose_eval.compute_ap()

        # —— 保存最佳模型 —— #
        # …（保存逻辑保持不变）…

        # —— WandB Epoch 日志 —— #
        # …（logging 保持不变）…

        if early_stop(avg_val_loss, model):
            print("Early stopping triggered")
            break

    # 最终模型保存
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model_final.pth"))
    destroy_process_group()



if __name__ == "__main__":
    args = arg_test()  # 或者 arg_real()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
    destroy_process_group()
