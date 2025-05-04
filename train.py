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
    ann_file = os.path.join(args.data_dir, "annotations", "person_keypoints_val2017.json")

    # 数据加载
    train_s = prepare_coco_dataset(args.data_dir, 'train', args.max_samples)
    val_s   = prepare_coco_dataset(args.data_dir, 'val',   args.max_samples)
    train_loader = DataLoader(COCODataset(train_s), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, collate_fn=collate_fn)
    val_loader   = DataLoader(COCODataset(val_s),   batch_size=args.batch_size,
                              shuffle=False,num_workers=args.num_workers,
                              pin_memory=True, collate_fn=collate_fn)

    # 模型等
    model = SegmentKeypointModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=args.epochs*len(train_loader),
        cycle_mult=1.0, max_lr=args.lr, min_lr=1e-6,
        warmup_steps=args.warmup_epochs*len(train_loader),
        gamma=1.0)
    scaler = GradScaler(enabled=args.use_fp16)
    early_stop = EarlyStopping(patience=args.patience, delta=args.min_delta)
    post_proc = PosePostProcessor()

    wandb.init(project=args.project_name, config=vars(args), entity=args.entity)

    best_ap = -float('inf')

    for epoch in range(1, args.epochs+1):
        # —— 训练 ——
        t0 = time.time()
        model.train(); train_loss=0.0
        pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for i, (imgs, masks, hm, paf, _,_,_) in enumerate(train_loader,1):
            bs = time.time()
            imgs,masks,hm,paf = imgs.to(device),masks.to(device),hm.to(device),paf.to(device)
            optimizer.zero_grad()
            if args.use_fp16:
                with autocast(device_type='cuda',enabled=True):
                    seg_pred,pose_pred = model(imgs)
                    loss = criterion(seg_pred,masks,pose_pred,hm,paf)
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
            else:
                seg_pred,pose_pred = model(imgs)
                loss = criterion(seg_pred,masks,pose_pred,hm,paf)
                loss.backward(); optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            bt = time.time()-bs
            lr = optimizer.param_groups[0]['lr']
            gn = sum(p.grad.norm(2).item()**2 for p in model.parameters() if p.grad is not None)**0.5
            wandb.log({'train/batch_loss':loss.item(),
                       'train/lr':lr,
                       'train/grad_norm':gn,
                       'train/batch_time':bt},
                      step=(epoch-1)*len(train_loader)+i)
            pbar.update(1); pbar.set_postfix({'loss':f"{loss.item():.4f}",
                                              'lr':f"{lr:.2e}",
                                              'bt':f"{bt:.2f}s"})
        pbar.close()
        avg_train = train_loss/len(train_loader)
        tr_time = time.time()-t0

        # —— 验证 ——
        v0=time.time()
        model.eval(); val_loss=0.0
        all_pred_masks,all_gt_masks=[],[]
        all_pred_kps,all_gt_kps=[],[]
        sample_seg_gts,sample_seg_preds=[],[]
        sample_pose_gts,sample_pose_preds=[],[]
        orig_sizes=[]

        with torch.no_grad():
            for imgs,masks,hm,paf,_,_,sizes in val_loader:
                imgs,masks,hm,paf = imgs.to(device),masks.to(device),hm.to(device),paf.to(device)
                with autocast(device_type='cuda',enabled=args.use_fp16):
                    seg_pred,pose_pred = model(imgs)
                    loss = criterion(seg_pred,masks,pose_pred,hm,paf)
                val_loss += loss.item()
                pm = (torch.sigmoid(seg_pred)>0.5).cpu().numpy()
                for b in range(pm.shape[0]):
                    h,w = masks[b,0].shape
                    m_uint8 = pm[b,0].astype(np.uint8)
                    m_res = cv2.resize(m_uint8,(int(w),int(h)),interpolation=cv2.INTER_NEAREST)
                    all_pred_masks.append(m_res); all_gt_masks.append(masks[b,0].cpu().numpy())
                    sample_seg_gts.append(masks[b]); sample_seg_preds.append(seg_pred[b])
                    sample_pose_gts.append(hm[b]); sample_pose_preds.append(pose_pred[b])
                    orig_sizes.append((h,w))
                # keypoints
                pred_kps = post_proc(pose_pred.cpu())
                gt_kps   = post_proc(hm.cpu())
                all_pred_kps.extend(pred_kps); all_gt_kps.extend(gt_kps)

        avg_val = val_loss/len(val_loader)
        vl_time = time.time()-v0

        # —— 评估指标 ——
        seg_eval = SegmentationEvaluator(num_classes=2)
        for pm,gt in zip(all_pred_masks,all_gt_masks):
            seg_eval.update(torch.from_numpy(pm).unsqueeze(0).unsqueeze(0),
                            torch.from_numpy(gt).unsqueeze(0).unsqueeze(0))
        seg_iou = seg_eval.compute_miou()

        pose_eval = PoseEvaluator(ann_file, COCO_KEYPOINT_NAMES)
        # 构造 preds list (img_id, heatmaps, scores)
        preds=[]
        for img_id, persons in enumerate(all_pred_kps):
            avg_score = np.mean([p[2] if len(p)>2 else 1.0 for person in persons for p in person])
            preds.append((img_id, persons, avg_score))
        pose_eval.update(preds, None)
        kp_acc = pose_eval.compute_ap()

        # —— 保存最佳模型 ——
        if kp_acc > best_ap:
            best_ap = kp_acc
            best_path = os.path.join(args.output_dir,"best_model.pth")
            torch.save(model.state_dict(),best_path)
            art = wandb.Artifact(name="best-keypoint-model", type="model",
                                 description=f"Epoch {epoch}, kp_acc={best_ap:.4f}")
            art.add_file(best_path)
            art.metadata = {"epoch":epoch,"seg_iou":seg_iou,"kp_acc":kp_acc}
            wandb.log_artifact(art, aliases=["best-keypoint"])
            wandb.run.summary["best_kp_acc"]=best_ap

        # —— WandB Epoch 日志 ——
        wandb.log({
            'train/avg_loss': avg_train,
            'val/avg_loss':   avg_val,
            'train/epoch_time': tr_time,
            'val/epoch_time':   vl_time,
            'val/seg_iou':      seg_iou,
            'val/kp_accuracy':  kp_acc,
            **{f"model/{n.replace('.', '/')}_hist": wandb.Histogram(p.detach().cpu().numpy())
               for n,p in model.named_parameters()}
        }, step=epoch)

        # 样本可视化
        log_samples(sample_seg_gts, sample_seg_preds,
                    sample_pose_gts, sample_pose_preds,
                    orig_sizes, epoch)

        if early_stop(avg_val, model):
            print("Early stopping triggered")
            break

    # 最终保存
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir,"model_final.pth"))
    destroy_process_group()


if __name__ == "__main__":
    args = arg_test()  # 或者 arg_real()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
    destroy_process_group()
