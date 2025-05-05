import os
import time
import random
import argparse
import zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image
import torchvision.transforms as T
import wandb
import cv2
from tqdm import tqdm
from scipy.ndimage import maximum_filter

from utils.visualization import visualize_coco_keypoints

# COCO骨骼拓扑结构: 17对关键点连接 (PAF 有 34 个通道)
COCO_PERSON_SKELETON = [(15,13), (13,11), (16,14), (14,12),
                        (5,11), (6,12), (5,7), (6,8),
                        (7,9), (8,10), (1,2), (0,1),
                        (0,2), (1,3), (2,4), (3,5), (4,6)]
NUM_LIMBS = len(COCO_PERSON_SKELETON)

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
    """Lightweight Pose Estimation Network (输出关键点热图和PAF)"""
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
        # PAF 向量场输出分支
        self.paf_conv = nn.Conv2d(base_channels, 2 * NUM_LIMBS, 1)
    def forward(self, x):
        x = self.stem(x)
        high = self.branch_high(x)
        low = self.down_conv(x)
        low = self.branch_low(low)
        low_up = F.interpolate(low, size=high.shape[2:], mode='bilinear', align_corners=False)
        feat = self.se(high + low_up)
        heatmap = self.final_conv(feat)
        paf = self.paf_conv(feat)
        # 输出 K 个关键点热图和 2L 个PAF特征图
        return heatmap, paf

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
        # 检查是否同时包含热图和PAF
        if isinstance(pred, (tuple, list)):
            # 分开关键点热图和PAF预测
            heat_pred, paf_pred = pred
            heat_target, paf_target = target
            # 关键点热图 MSE 损失 + OHKM
            N, K, H, W = heat_pred.shape
            heat_loss_all = self.mse(heat_pred, heat_target)
            per_channel = ((heat_pred - heat_target)**2).view(N, K, -1).mean(dim=2)
            topk_loss = torch.stack([
                v.topk(self.ohkm_k, largest=True)[0].mean() for v in per_channel
            ]).mean()
            heat_loss = heat_loss_all + topk_loss
            # PAF 向量场 MSE 损失
            paf_loss = self.mse(paf_pred, paf_target)
            return heat_loss + paf_loss
        else:
            # 仅关键点热图损失 (无 PAF 分支)
            N, K, H, W = pred.shape
            loss_all = self.mse(pred, target)
            per_channel = ((pred - target)**2).view(N, K, -1).mean(dim=2)
            topk_loss = torch.stack([
                v.topk(self.ohkm_k, largest=True)[0].mean() for v in per_channel
            ]).mean()
            return loss_all + topk_loss

# --------------------------------------------------
# Utility: Ensure COCO dataset integrity and download
# --------------------------------------------------
def ensure_coco_data(root):
    """
    检查 COCO 数据集是否完整；如缺失则自动下载并解压，下载时显示 tqdm 进度条。
    """
    urls = {
        'train2017.zip': 'http://images.cocodataset.org/zips/train2017.zip',
        'val2017.zip':   'http://images.cocodataset.org/zips/val2017.zip',
        'annotations.zip': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    }
    os.makedirs(root, exist_ok=True)
    def _download_with_progress(filename, url):
        zip_path = os.path.join(root, filename)
        # 如果文件不存在或太小则重新下载
        if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 100:
            # Streaming download with progress bar
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total = int(r.headers.get('content-length', 0))
                with open(zip_path, 'wb') as f, tqdm(
                    desc=f"Downloading {filename}",
                    total=total,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))
        return zip_path
    def _extract_zip(zip_path):
        print(f"Extracting {os.path.basename(zip_path)} to {root}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            members = zf.infolist()
            # Use tqdm to show progress over the members list
            for member in tqdm(members, desc="Extracting", unit="file"):
                zf.extract(member, root)
        print(f"Extraction {os.path.basename(zip_path)} complete.")
    # 检查并下载/解压数据集
    expected_dirs = ['train2017', 'val2017', 'annotations']
    for zip_name, url in urls.items():
        target_dir = zip_name.replace('.zip', '')
        if not os.path.isdir(os.path.join(root, target_dir)):
            zip_path = _download_with_progress(zip_name, url)
            _extract_zip(zip_path)

# -----------------------
# COCO Dataset
# -----------------------
class COCOPoseDataset(torch.utils.data.Dataset):
    """
    COCO Pose Dataset: 自动检查 & 下载数据集
    返回全图像及对应关键点热图和PAF张量
    """
    def __init__(self, root, ann_file, img_folder,
                 img_size=(256,192), hm_size=(64,48), sigma=2):
        # 确保数据完整
        ensure_coco_data(root)
        self.coco = COCO(ann_file)
        all_img_ids = self.coco.getImgIds(catIds=[1])  # 所有包含人的图像ID
        self.img_ids = []
        for img_id in all_img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=[1], iscrowd=None)
            anns = self.coco.loadAnns(ann_ids)
            # 如果此图像中至少有一个人有标注关键点
            if any(ann['num_keypoints'] > 0 for ann in anns):
                self.img_ids.append(img_id)
        self.root = root
        self.img_folder = img_folder
        self.img_size = img_size
        self.hm_size = hm_size
        self.sigma = sigma
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        # 将样本列表定义为图像ID列表（方便兼容旧接口）
        self.samples = self.img_ids

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        # 获取图像及注释
        img_id = self.img_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=[1], iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)
        # 加载原始图像并调整为统一大小
        img_info = self.coco.loadImgs([img_id])[0]
        path = os.path.join(self.root, self.img_folder, img_info['file_name'])
        img = Image.open(path).convert('RGB')
        orig_w, orig_h = img.width, img.height
        img = img.resize((self.img_size[1], self.img_size[0]))
        # 初始化关键点热图和 PAF 张量
        num_kp = 17
        L = len(COCO_PERSON_SKELETON)
        hm = np.zeros((num_kp, self.hm_size[0], self.hm_size[1]), dtype=np.float32)
        paf_x = np.zeros((L, self.hm_size[0], self.hm_size[1]), dtype=np.float32)
        paf_y = np.zeros((L, self.hm_size[0], self.hm_size[1]), dtype=np.float32)
        count = np.zeros((L, self.hm_size[0], self.hm_size[1]), dtype=np.int32)
        # 遍历图像中每个人的注释
        for ann in anns:
            if ann['num_keypoints'] == 0:
                continue  # 跳过未标注关键点的（例如 iscrowd）人员
            keypoints = np.array(ann['keypoints']).reshape(-1, 3)
            # 绘制关键点热图
            for k, (x, y, v) in enumerate(keypoints):
                if v > 0:  # v=2或1 表示可见/不可见但标注
                    hm_x = x * (self.hm_size[1] / float(orig_w))
                    hm_y = y * (self.hm_size[0] / float(orig_h))
                    ul = [int(hm_x - 3*self.sigma), int(hm_y - 3*self.sigma)]
                    br = [int(hm_x + 3*self.sigma), int(hm_y + 3*self.sigma)]
                    for yy in range(max(0, ul[1]), min(self.hm_size[0], br[1] + 1)):
                        for xx in range(max(0, ul[0]), min(self.hm_size[1], br[0] + 1)):
                            d2 = (xx - hm_x)**2 + (yy - hm_y)**2
                            if d2 <= 9 * (self.sigma**2):
                                hm[k, yy, xx] = max(hm[k, yy, xx], np.exp(-d2 / (2 * self.sigma**2)))
            # 绘制每条骨骼连接的 PAF
            for c, (a, b) in enumerate(COCO_PERSON_SKELETON):
                xa, ya, va = keypoints[a]
                xb, yb, vb = keypoints[b]
                if va > 0 and vb > 0:
                    xa_hm = xa * (self.hm_size[1] / float(orig_w))
                    ya_hm = ya * (self.hm_size[0] / float(orig_h))
                    xb_hm = xb * (self.hm_size[1] / float(orig_w))
                    yb_hm = yb * (self.hm_size[0] / float(orig_h))
                    dx = xb_hm - xa_hm
                    dy = yb_hm - ya_hm
                    norm = np.sqrt(dx**2 + dy**2) + 1e-8
                    vx = dx / norm
                    vy = dy / norm
                    # 计算连接线段覆盖的边界框范围
                    min_x = int(max(0, min(xa_hm, xb_hm) - 1))
                    max_x = int(min(self.hm_size[1] - 1, max(xa_hm, xb_hm) + 1))
                    min_y = int(max(0, min(ya_hm, yb_hm) - 1))
                    max_y = int(min(self.hm_size[0] - 1, max(ya_hm, yb_hm) + 1))
                    for yy in range(min_y, max_y + 1):
                        for xx in range(min_x, max_x + 1):
                            # 计算像素到线段的垂直距离
                            t = ((xx - xa_hm) * dx + (yy - ya_hm) * dy) / (norm**2)
                            if t < 0 or t > 1:
                                continue
                            proj_x = xa_hm + t * dx
                            proj_y = ya_hm + t * dy
                            dist = np.sqrt((xx - proj_x)**2 + (yy - proj_y)**2)
                            if dist <= 1:
                                paf_x[c, yy, xx] += vx
                                paf_y[c, yy, xx] += vy
                                count[c, yy, xx] += 1
        # 平均处理重叠区域的 PAF
        for c in range(L):
            mask = count[c] > 0
            if np.any(mask):
                paf_x[c, mask] /= count[c, mask]
                paf_y[c, mask] /= count[c, mask]
        # 合并 PAF 的 X/Y 分量通道
        paf = np.zeros((2*L, self.hm_size[0], self.hm_size[1]), dtype=np.float32)
        for c in range(L):
            paf[2*c] = paf_x[c]
            paf[2*c+1] = paf_y[c]
        # 图像张量归一化
        img_tensor = self.transform(img)
        return img_tensor, torch.from_numpy(hm), torch.from_numpy(paf)

# -----------------------
# Evaluation and Visualization
# -----------------------
def evaluate(model, val_loader, device):
    """
    Evaluate model on COCO val set; return mean AP, AP50, WandB images list
    """
    model.eval()
    coco_gt = val_loader.dataset.coco
    results, img_ids, vis_list = [], [], []
    n_vis = getattr(wandb.config, 'n_vis', 3)
    vis_ids = random.sample(val_loader.dataset.img_ids, min(n_vis, len(val_loader.dataset.img_ids)))
    idx_offset = 0

    with torch.no_grad():
        for imgs, _, _ in val_loader:
            imgs = imgs.to(device)
            heat_pred, paf_pred = model(imgs)
            B, H, W = imgs.size(0), heat_pred.shape[2], heat_pred.shape[3]

            for i in range(B):
                img_id = val_loader.dataset.img_ids[idx_offset + i]
                img_ids.append(img_id)
                heat = heat_pred[i].cpu().numpy()
                paf = paf_pred[i].cpu().numpy()

                # 1. Peak detection
                all_peaks = []
                thresh = 0.1
                for k in range(heat.shape[0]):
                    hmap = heat[k]
                    peaks = ((hmap == maximum_filter(hmap, size=3)) & (hmap > thresh))
                    coords = np.argwhere(peaks)
                    all_peaks.append([(float(x), float(y), float(hmap[y,x])) for y,x in coords])

                # 2. Connection candidates & greedy
                connection_candidates = {c: [] for c in range(len(COCO_PERSON_SKELETON))}
                for c,(a,b) in enumerate(COCO_PERSON_SKELETON):
                    candA, candB = all_peaks[a], all_peaks[b]
                    if not candA or not candB: continue
                    paf_x, paf_y = paf[2*c], paf[2*c+1]
                    for ia,(ax,ay,sa) in enumerate(candA):
                        for ib,(bx,by,sb) in enumerate(candB):
                            dx,dy=bx-ax,by-ay; norm=np.hypot(dx,dy)+1e-8
                            vx,vy=dx/norm,dy/norm
                            samples=10
                            xs=np.linspace(ax,bx,samples).astype(int)
                            ys=np.linspace(ay,by,samples).astype(int)
                            vec_scores=[np.dot([paf_x[yy,xx],paf_y[yy,xx]],[vx,vy])
                                         for xx,yy in zip(xs,ys)
                                         if 0<=xx<W and 0<=yy<H]
                            if vec_scores and np.mean(vec_scores)>0 and np.mean(np.array(vec_scores)>0.05)>0.8:
                                connection_candidates[c].append((ia,ib,np.mean(vec_scores)+0.5*(sa+sb)))
                    connection_candidates[c].sort(key=lambda x: x[2], reverse=True)
                    usedA,usedB,conns=set(),set(),[]
                    for ia,ib,_ in connection_candidates[c]:
                        if ia not in usedA and ib not in usedB:
                            conns.append((ia,ib)); usedA.add(ia); usedB.add(ib)
                    connection_candidates[c] = conns

                # 3. Assemble persons
                persons = []
                for c,(a,b) in enumerate(COCO_PERSON_SKELETON):
                    for ia,ib in connection_candidates[c]:
                        placed=False
                        for p in persons:
                            if (a in p and p[a]==ia) or (b in p and p[b]==ib):
                                p[a]=ia; p[b]=ib; placed=True; break
                        if not placed: persons.append({a:ia,b:ib})
                for k,peaks in enumerate(all_peaks):
                    for idx in range(len(peaks)):
                        if not any(p.get(k)==idx for p in persons):
                            persons.append({k:idx})

                # 4. COCO-format results
                orig = coco_gt.loadImgs([img_id])[0]
                for p in persons:
                    kps_list,sc_list=[],[]
                    for k in range(len(all_peaks)):
                        if k in p:
                            x,y,sc=all_peaks[k][p[k]]
                            x *= (orig['width']/W); y *= (orig['height']/H)
                        else: x,y,sc=0.0,0.0,0.0
                        kps_list.extend([float(x),float(y),float(sc)]); sc_list.append(sc)
                    results.append({'image_id':img_id,'category_id':1,'keypoints':kps_list,'score':float(np.mean(sc_list))})

                # 5. Visualization
                if img_id in vis_ids:
                    img_info = coco_gt.loadImgs([img_id])[0]
                    img_path = f"{val_loader.dataset.root}/{val_loader.dataset.img_folder}/{img_info['file_name']}"
                    orig_img = cv2.imread(img_path) or np.zeros((img_info['height'],img_info['width'],3),dtype=np.uint8)
                    tw,th = val_loader.dataset.img_size[1], val_loader.dataset.img_size[0]

                    # GT visualize
                    gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[img_id],catIds=[1],iscrowd=None))
                    vis_img = visualize_coco_keypoints(orig_img, gt_anns, COCO_PERSON_SKELETON, (tw,th), (0,255,0), (0,255,0))

                    # Prepare pred anns for visualization
                    pred_anns = []
                    for p in persons:
                        kplist=[];num_kp=0
                        for k in range(len(all_peaks)):
                            if k in p:
                                x,y,sc = all_peaks[k][p[k]]
                                x *= (orig['width']/W); y *= (orig['height']/H)
                                v = 2 if sc>0 else 0
                                num_kp+=1
                            else: x,y,v=0.0,0.0,0
                            kplist.extend([x,y,v])
                        pred_anns.append({'keypoints':kplist,'num_keypoints':num_kp})

                    # Pred visualize on same image
                    vis_img = visualize_coco_keypoints(vis_img, pred_anns, COCO_PERSON_SKELETON, (tw,th), (0,0,255), (0,0,255))

                    vis_list.append(
                        wandb.Image(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB),
                                   caption=f"Image {img_id} – GT(green) vs Pred(red)")
                    )

            idx_offset += B

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='keypoints')
    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
    return coco_eval.stats[0], coco_eval.stats[1], vis_list
# -----------------------
# Training Loop
# -----------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0
    for batch_idx, (imgs, heatmaps, pafs) in enumerate(tqdm(loader, desc="Training:", leave=False)):
        imgs = imgs.to(device)
        heatmaps = heatmaps.to(device)
        pafs = pafs.to(device)
        optimizer.zero_grad()
        heat_pred, paf_pred = model(imgs)
        loss = criterion((heat_pred, paf_pred), (heatmaps, pafs))
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

    return epoch_loss / len(loader.dataset)

# -----------------------
# Main
# -----------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="轻量化多人骨骼姿态检测训练脚本")
    parser.add_argument('--data_root', type=str, default='run/data', help='Path to COCO dataset root directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate for optimizer')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--img_h', type=int, default=256, help='Input image height')
    parser.add_argument('--img_w', type=int, default=256, help='Input image width')
    parser.add_argument('--hm_h', type=int, default=64, help='Output heatmap height')
    parser.add_argument('--hm_w', type=int, default=64, help='Output heatmap width')
    parser.add_argument('--sigma', type=int, default=2, help='Sigma for Gaussian heatmap generation')
    parser.add_argument('--ohkm_k', type=int, default=8, help='Number of hardest keypoints to mine for OHKM loss')
    parser.add_argument('--n_vis', type=int, default=3, help='Number of validation samples to visualize per epoch')

    parser.add_argument('--num_workers_train', type=int, default=16,help = 'Number of workers for training DataLoader (B200: 28 vCPU)')
    parser.add_argument('--num_workers_val', type=int, default=8,help = 'Number of workers for validation DataLoader')

    args = parser.parse_args()

    os.makedirs('run/models', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize WandB
    wandb.init(project='final', entity="joint_angle",config=vars(args))
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
    print(f"Train Sample: {len(train_ds)}")
    print(f"Val Sample: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers_train,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers_val,
        pin_memory=True
    )

    best_ap = 0.0

    for epoch in range(1, config.epochs + 1):
        start_time = time.time()
        # --- Training ---
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        # 更新学习率调度器
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        # --- Validation ---
        val_ap, val_acc, val_vis_images = evaluate(model, val_loader, device)
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
        }, commit=False)
        # Log parameter & gradient histograms
        for name, param in model.named_parameters():
            wandb.log({f"param/{name}": wandb.Histogram(param.detach().cpu().numpy()),
                       f"grad/{name}": wandb.Histogram(param.grad.detach().cpu().numpy() if param.grad is not None else np.zeros(1))},commit=False)
        wandb.log({},  commit=True)
        # Save checkpoint and upload as WandB Artifact
        ckpt_path = f'run/models/epoch{epoch}.pth'
        torch.save(model.state_dict(), ckpt_path)
        artifact = wandb.Artifact(f'lite-pose-model-epoch{epoch}', type='model')
        artifact.add_file(ckpt_path)
        wandb.log_artifact(artifact)
        # 可视化部分验证结果
        if len(val_vis_images) > 0:
            wandb.log({'val_examples': val_vis_images})
        # Save best model
        if val_ap > best_ap:
            best_ap = val_ap
            torch.save(model.state_dict(), 'run/models/best_model.pth')
    wandb.finish()


