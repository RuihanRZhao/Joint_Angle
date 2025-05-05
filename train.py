import os
import time
import random
import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import wandb

from utils.coco import prepare_coco_dataset, COCODataset, collate_fn
from utils.visualization import SKELETON, COCO_KEYPOINT_NAMES
from models.SegKP_Model import SegmentKeypointModel, PosePostProcessor
from utils.loss import criterion
from utils.Evaluator import SegmentationEvaluator, PoseEvaluator
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from early_stopping_pytorch import EarlyStopping
from config import arg_real, arg_test


def create_mask_visual(mask):
    vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
    vis[mask > 0] = [255, 0, 0]
    return vis


def create_pose_visual(kps_list, image_size, skeleton):
    h, w = image_size
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    for person in kps_list:
        for (a, b) in skeleton:
            x1, y1 = person[a]
            x2, y2 = person[b]
            if x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0:
                cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        for (x, y) in person:
            if x >= 0 and y >= 0:
                cv2.circle(vis, (int(x), int(y)), 3, (0,0,255), -1)
    return vis


def log_samples(seg_gts, seg_preds, pose_gts, pose_preds, original_sizes, epoch):
    post_proc = PosePostProcessor()
    for i in range(min(len(seg_gts), 2)):
        # Segmentation GT & Pred
        gt_mask = seg_gts[i].cpu().numpy().squeeze().astype(np.uint8)
        gt_vis = create_mask_visual(gt_mask)
        pred_logit = seg_preds[i].cpu().detach().numpy().squeeze()
        pred_mask = (torch.sigmoid(torch.from_numpy(pred_logit).unsqueeze(0)) > 0.5).numpy()[0]
        pred_vis = create_mask_visual(pred_mask.astype(np.uint8))
        # Pose GT & Pred
        orig_h, orig_w = original_sizes[i]
        gt_hm = cv2.resize(pose_gts[i].cpu().numpy().transpose(1,2,0), (orig_w, orig_h))
        gt_kps = post_proc(torch.tensor(gt_hm).permute(2,0,1).unsqueeze(0))[0]
        gt_pose = create_pose_visual(gt_kps, (orig_h, orig_w), SKELETON)
        pred_hm = F.interpolate(pose_preds[i].unsqueeze(0), size=(orig_h, orig_w),
                                mode='bilinear', align_corners=False)[0]
        pred_kps = post_proc(pred_hm.unsqueeze(0))[0]
        pred_pose = create_pose_visual(pred_kps, (orig_h, orig_w), SKELETON)
        # Log each image
        wandb.log({
            f"val/gt_seg_epoch{epoch}_sample{i}": wandb.Image(gt_vis),
            f"val/pred_seg_epoch{epoch}_sample{i}": wandb.Image(pred_vis),
            f"val/gt_pose_epoch{epoch}_sample{i}": wandb.Image(gt_pose),
            f"val/pred_pose_epoch{epoch}_sample{i}": wandb.Image(pred_pose)
        })


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Data loaders
    train_samples = prepare_coco_dataset(args.data_dir, split='train', max_samples=args.max_samples)
    val_samples   = prepare_coco_dataset(args.data_dir, split='val',   max_samples=args.max_samples)
    train_loader = DataLoader(COCODataset(train_samples), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    val_loader   = DataLoader(COCODataset(val_samples),   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    # Model, optimizer, scheduler, fp16 scaler, early stopping
    model      = SegmentKeypointModel().to(device)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler  = CosineAnnealingWarmupRestarts(
        optimizer, first_cycle_steps=args.epochs*len(train_loader),
        cycle_mult=1.0, max_lr=args.lr, min_lr=1e-6,
        warmup_steps=args.warmup_epochs*len(train_loader), gamma=1.0)
    scaler     = GradScaler(device="cuda", enabled=args.use_fp16)
    early_stop = EarlyStopping(patience=args.patience, delta=args.min_delta)
    post_proc  = PosePostProcessor()

    # WandB init
    wandb.init(project=args.project_name, config=vars(args), entity=args.entity)
    best_ap = -1.0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        epoch_train_start = time.time()
        pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for batch_idx, (imgs, masks, hm, paf, _, _, _) in enumerate(train_loader, start=1):
            imgs, masks, hm, paf = imgs.to(device), masks.to(device), hm.to(device), paf.to(device)
            optimizer.zero_grad()
            if args.use_fp16:
                with autocast(device_type="cuda"):
                    seg_pred, pose_pred = model(imgs)
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
            # grad norm
            total_norm_sqr = torch.zeros(1, device=device)
            for p in model.parameters():
                if p.grad is not None:
                    total_norm_sqr += p.grad.norm(2) ** 2
            grad_norm = torch.sqrt(total_norm_sqr).item()
            # Log batch
            global_step += 1
            wandb.log({
                'train/loss':      loss.item(),
                'train/lr':        optimizer.param_groups[0]['lr'],
                'train/grad_norm': grad_norm
            }, step=global_step)
            pbar.update(1)
        pbar.close()
        avg_train_loss = train_loss / len(train_loader)
        train_time = time.time() - epoch_train_start

        # Validation
        model.eval()
        val_loss = 0.0
        epoch_val_start = time.time()
        all_pred_masks, all_gt_masks = [], []
        sample_seg_gts, sample_seg_preds = [], []
        sample_pose_gts, sample_pose_preds = [], []
        orig_sizes = []
        paths_list = []
        with torch.no_grad():
            for imgs, masks, hm, paf, _, _, paths in val_loader:
                imgs, masks, hm, paf = imgs.to(device), masks.to(device), hm.to(device), paf.to(device)
                seg_pred, pose_pred = model(imgs)
                loss = criterion(seg_pred, masks, pose_pred, hm, paf)
                val_loss += loss.item()
                pm = (torch.sigmoid(seg_pred) > 0.5).cpu().numpy().astype(np.uint8)
                for b in range(pm.shape[0]):
                    h_i, w_i = masks.shape[2], masks.shape[3]
                    m_res = cv2.resize(pm[b,0], (w_i, h_i), interpolation=cv2.INTER_NEAREST)
                    all_pred_masks.append(m_res)
                    all_gt_masks.append(masks[b,0].cpu().numpy())
                    sample_seg_gts.append(masks[b])
                    sample_seg_preds.append(seg_pred[b])
                    sample_pose_gts.append(hm[b])
                    sample_pose_preds.append(pose_pred[b])
                    orig_sizes.append((h_i, w_i))
                paths_list.extend(paths)
        avg_val_loss = val_loss / len(val_loader)
        val_time = time.time() - epoch_val_start

        # Seg IoU
        seg_eval = SegmentationEvaluator(num_classes=2)
        for pm_arr, gt_arr in zip(all_pred_masks, all_gt_masks):
            seg_eval.update(torch.from_numpy(pm_arr[None,None,:,:]), torch.from_numpy(gt_arr[None,None,:,:]))
        seg_iou = seg_eval.compute_miou()

        # Pose AP
        batch_preds = []
        for idx, hm_arr in enumerate(sample_pose_preds):
            fname = os.path.splitext(os.path.basename(paths_list[idx]))[0]
            try:
                image_id = int(fname)
            except ValueError:
                continue
            persons = post_proc(hm_arr.unsqueeze(0))[0]
            if not persons:
                continue
            scores = [1.0] * len(persons)
            batch_preds.append((image_id, persons, scores))
        if batch_preds:
            pose_eval = PoseEvaluator(
                os.path.join(args.data_dir, "annotations/person_keypoints_val2017.json"),
                COCO_KEYPOINT_NAMES
            )
            pose_eval.update(batch_preds, None)
            pose_ap = pose_eval.compute_ap()
        else:
            pose_ap = 0.0
            print("Warning: No valid pose detections available; setting pose_ap=0.0")

        # Save best model
        if pose_ap > best_ap:
            best_ap = pose_ap
            os.makedirs(os.path.join(args.output_dir, "models"), exist_ok=True)
            best_path = os.path.join(args.output_dir, "models", f"best_keypoint_model_epoch{epoch}.pth")
            torch.save(model.state_dict(), best_path)
            art = wandb.Artifact("best-keypoint-model", type="model",
                                 description=f"Epoch {epoch}, pose_ap={best_ap:.4f}")
            art.add_file(best_path)
            wandb.log_artifact(art)
            wandb.run.summary["best_pose_ap"] = best_ap

        # Epoch-level log (no step)
        wandb.log({
            'train/avg_loss':   avg_train_loss,
            'val/avg_loss':     avg_val_loss,
            'train/epoch_time': train_time,
            'val/epoch_time':   val_time,
            'val/seg_iou':      seg_iou,
            'val/pose_ap':      pose_ap,
        })

        # Visualize samples
        log_samples(sample_seg_gts, sample_seg_preds,
                    sample_pose_gts, sample_pose_preds,
                    orig_sizes, epoch)

        # Early stopping
        if early_stop(avg_val_loss, model):
            print("Early stopping triggered")
            break

    # Save final
    os.makedirs(os.path.join(args.output_dir, "models"), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "models", "model_final.pth"))

    # Destroy process group if distributed
    if args.dist and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    args = arg_real()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
