import os
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image
import torchvision.transforms as T
import wandb
import cv2
from tqdm import tqdm

# -----------------------
# Model Definitions
# -----------------------
class DepthwiseSeparableConv(nn.Module):
    """Depthwise + Pointwise Convolution"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depth = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding,
                               groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.point = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.relu(self.bn1(self.depth(x)))
        x = self.relu(self.bn2(self.point(x)))
        return x

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = self.relu(self.fc1(w))
        w = self.sigmoid(self.fc2(w))
        return x * w

class MultiPoseNet(nn.Module):
    """Lightweight Pose Estimation Network"""
    def __init__(self, num_keypoints=17, base_channels=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, 7, 2, 3, bias=False),
            nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True)
        )
        self.branch_high = nn.Sequential(
            DepthwiseSeparableConv(base_channels, base_channels),
            DepthwiseSeparableConv(base_channels, base_channels)
        )
        self.down_conv = DepthwiseSeparableConv(base_channels, base_channels*2, stride=2)
        self.branch_low = nn.Sequential(
            DepthwiseSeparableConv(base_channels*2, base_channels*2),
            DepthwiseSeparableConv(base_channels*2, base_channels)
        )
        self.se = SEBlock(base_channels)
        self.final_conv = nn.Conv2d(base_channels, num_keypoints, 1)

    def forward(self, x):
        x = self.stem(x)
        high = self.branch_high(x)
        low = self.down_conv(x)
        low = self.branch_low(low)
        low_up = F.interpolate(low, size=high.shape[2:], mode='bilinear', align_corners=False)
        feat = self.se(high + low_up)
        heatmap = self.final_conv(feat)
        return heatmap

# -----------------------
# Loss Function
# -----------------------
class PoseLoss(nn.Module):
    """MSE + Online Hard Keypoint Mining"""
    def __init__(self, ohkm_k=8):
        super().__init__()
        self.ohkm_k = ohkm_k
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, pred, target):
        N, K, H, W = pred.shape
        loss_all = self.mse(pred, target)
        per_channel = ((pred - target)**2).view(N, K, -1).mean(dim=2)
        topk_loss = torch.stack([
            v.topk(self.ohkm_k, largest=True)[0].mean() for v in per_channel
        ]).mean()
        return loss_all + topk_loss

# -----------------------
# COCO Dataset
# -----------------------
class COCOPoseDataset(torch.utils.data.Dataset):
    def __init__(self, root, ann_file, img_folder,
                 img_size=(256,192), hm_size=(64,48), sigma=2):
        self.coco = COCO(ann_file)
        img_ids = self.coco.getImgIds(catIds=[1])
        self.samples = []
        for img_id in img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=[1], iscrowd=None)
            for aid in ann_ids:
                ann = self.coco.loadAnns([aid])[0]
                if ann['num_keypoints'] > 0:
                    self.samples.append((img_id, aid))
        self.root = root
        self.img_folder = img_folder
        self.img_size = img_size
        self.hm_size = hm_size
        self.sigma = sigma
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, ann_id = self.samples[idx]
        ann = self.coco.loadAnns([ann_id])[0]
        img_info = self.coco.loadImgs([img_id])[0]
        path = os.path.join(self.root, self.img_folder, img_info['file_name'])
        img = Image.open(path).convert('RGB')
        x,y,w,h = ann['bbox']
        # Expand bbox by 1.2
        c_x, c_y = x + w/2, y + h/2
        w *= 1.2; h *= 1.2
        x1 = max(0, int(c_x - w/2)); y1 = max(0, int(c_y - h/2))
        x2 = min(img.width, int(c_x + w/2)); y2 = min(img.height, int(c_y + h/2))
        img = img.crop((x1,y1,x2,y2)).resize((self.img_size[1], self.img_size[0]))
        keypoints = np.array(ann['keypoints']).reshape(-1,3)
        hm = np.zeros((len(keypoints), *self.hm_size), dtype=np.float32)
        for i, (px,py,v) in enumerate(keypoints):
            if v>0:
                hm_x = px-x1; hm_y = py-y1
                hm_x *= self.hm_size[1]/self.img_size[1]
                hm_y *= self.hm_size[0]/self.img_size[0]
                ul = [int(hm_x-3*self.sigma), int(hm_y-3*self.sigma)]
                br = [int(hm_x+3*self.sigma), int(hm_y+3*self.sigma)]
                for yy in range(max(0,ul[1]), min(self.hm_size[0], br[1]+1)):
                    for xx in range(max(0,ul[0]), min(self.hm_size[1], br[0]+1)):
                        d2 = (xx-hm_x)**2 + (yy-hm_y)**2
                        if d2 <= 9*self.sigma*self.sigma:
                            hm[i,yy,xx] = max(hm[i,yy,xx], np.exp(-d2/(2*self.sigma**2)))
        img = self.transform(img)
        return img, torch.from_numpy(hm)

# -----------------------
# Evaluation and Visualization
# -----------------------
def evaluate(model, val_loader, device):
    """
    Evaluate the model on the COCO validation set and compute keypoint AP & accuracy.

    Args:
        model (nn.Module): The pose estimation model.
        val_loader (DataLoader): DataLoader for the validation set.
        device (torch.device): Device to run inference on.

    Returns:
        tuple: (mean Average Precision (AP), keypoint accuracy)
    """
    model.eval()
    coco_gt = val_loader.dataset.coco
    results = []
    img_ids = []

    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)  # [N, K, H, W]
            N, K, H, W = outputs.shape

            # For each sample in batch
            for i in range(N):
                # Retrieve original image ID
                img_id, _ = val_loader.dataset.samples[i]
                img_ids.append(img_id)

                # Convert heatmaps to keypoint predictions
                heatmaps = outputs[i].cpu().numpy()
                # For each keypoint channel, find the argmax
                keypoints = []
                for k in range(K):
                    hmap = heatmaps[k]
                    y, x = np.unravel_index(hmap.argmax(), hmap.shape)
                    # Scale back to original image coordinates
                    scale_x = val_loader.dataset.img_size[1] / float(W)
                    scale_y = val_loader.dataset.img_size[0] / float(H)
                    keypoints.append((x * scale_x, y * scale_y, hmap.max()))

                # Build detection entry for COCO format
                results.append({
                    "image_id": img_id,
                    "category_id": 1,  # person
                    "keypoints": [coord for kp in keypoints for coord in kp],
                    "score": float(np.mean([kp[2] for kp in keypoints]))
                })

    # Load results into COCOeval
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='keypoints')
    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extract mean AP (OKS) and a simple accuracy metric (e.g., AP@0.5)
    mean_ap = coco_eval.stats[0]  # AP (IoU=0.50:0.95)
    acc_50 = coco_eval.stats[2]  # AP at OKS=0.5

    return mean_ap, acc_50


# -----------------------
# Training Loop
# -----------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0
    for batch_idx, (imgs, targets) in enumerate(tqdm(loader, desc="Training:", leave=False)):
        imgs = imgs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, targets)
        loss.backward()

        # 记录梯度范数
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        optimizer.step()
        epoch_loss += loss.item() * imgs.size(0)

        # 每 N 个 batch 记录一次
        if batch_idx % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({
                "batch/train_loss": loss.item(),
                "batch/grad_norm": total_norm,
                "batch/lr": current_lr,
                "batch_idx": batch_idx
            })

    return epoch_loss / len(loader.dataset)

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='run/data')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--img_h', type=int, default=256)
    parser.add_argument('--img_w', type=int, default=192)
    parser.add_argument('--hm_h', type=int, default=64)
    parser.add_argument('--hm_w', type=int, default=48)
    parser.add_argument('--sigma', type=int, default=2)
    parser.add_argument('--ohkm_k', type=int, default=8)
    parser.add_argument('--n_vis', type=int, default=3)
    args = parser.parse_args()

    os.makedirs('run/models', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize WandB
    wandb.init(project='final', config=vars(args))
    config = wandb.config

    # Model, loss, optimizer, scheduler
    model = MultiPoseNet(num_keypoints=17, base_channels=64).to(device)
    criterion = PoseLoss(ohkm_k=config.ohkm_k)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # Data
    train_ds = COCOPoseDataset(
        root=args.data_root,
        ann_file=os.path.join(args.data_root, 'annotations/person_keypoints_train2017.json'),
        img_folder='train2017',
        img_size=(args.img_h, args.img_w),
        hm_size=(args.hm_h, args.hm_w),
        sigma=args.sigma
    )
    val_ds = COCOPoseDataset(
        root=args.data_root,
        ann_file=os.path.join(args.data_root, 'annotations/person_keypoints_val2017.json'),
        img_folder='val2017',
        img_size=(args.img_h, args.img_w),
        hm_size=(args.hm_h, args.hm_w),
        sigma=args.sigma
    )
    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size,
                            shuffle=False, num_workers=2)

    best_ap = 0.0

    for epoch in range(1, config.epochs + 1):
        start_time = time.time()

        # --- Training ---
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # --- Validation ---
        val_ap, val_acc = evaluate(model, val_loader, device)

        epoch_time = time.time() - start_time

        # Log epoch-level metrics
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_ap': val_ap,
            'val_acc': val_acc,
            'epoch_time': epoch_time,
            'lr': current_lr,
            'gpu_mem_used': torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
        })

        # Log parameter & gradient histograms
        for name, param in model.named_parameters():
            wandb.log({f"param/{name}": wandb.Histogram(param.detach().cpu().numpy()),
                       f"grad/{name}": wandb.Histogram(param.grad.detach().cpu().numpy() if param.grad is not None else np.zeros(1))},
                      commit=False)
        wandb.log({}, commit=True)

        # Save checkpoint and upload as WandB Artifact
        ckpt_path = f'checkpoints/epoch{epoch}.pth'
        torch.save(model.state_dict(), ckpt_path)
        artifact = wandb.Artifact(f'lite-pose-model-epoch{epoch}', type='model')
        artifact.add_file(ckpt_path)
        wandb.log_artifact(artifact)

        # Visualize a few validation examples
        vis_list = []
        for _ in range(config.n_vis):
            idx = random.randint(0, len(val_ds) - 1)
            img, heat_gt = val_ds[idx]
            inp = img.unsqueeze(0).to(device)
            with torch.no_grad():
                heat_pred = model(inp)[0].cpu().numpy()

            img_np = (img.permute(1,2,0).numpy() * np.array([0.229,0.224,0.225]) +
                      np.array([0.485,0.456,0.406]))
            img_np = np.clip(img_np, 0, 1)

            # Extract GT / pred keypoints for visualization
            gt_pts = np.stack(np.where(
                heat_gt.numpy() == heat_gt.max(dim=-1, keepdim=True)[0].max(dim=-2)[0]
            ))[::-1].T
            pred_pts = np.stack(np.where(
                heat_pred == heat_pred.max(axis=1, keepdims=True).max(axis=2)[0]
            ))[::-1].T

            img_vis = img_np.copy()
            for (x,y) in gt_pts:
                cv2.circle(img_vis, (int(x), int(y)), 2, (0,1,0), -1)  # GT: green
            for (x,y) in pred_pts:
                cv2.circle(img_vis, (int(x), int(y)), 2, (1,0,0), -1)  # Pred: red

            vis_list.append(wandb.Image(img_vis,
                                        caption=f"GT vs Pred (epoch {epoch})"))
        wandb.log({'val_examples': vis_list})

        # Save best model
        if val_ap > best_ap:
            best_ap = val_ap
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')

    wandb.finish()


if __name__ == '__main__':
    main()