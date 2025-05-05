import math
import os
import random
import numpy as np
import torch
from pycocotools.coco import COCO
from PIL import Image
import torchvision.transforms as T
import cv2
from torch.utils.data import Dataset
from typing import Tuple, List
from tqdm import tqdm
import requests

COCO_PERSON_SKELETON: List[Tuple[int, int]] = [
    (15, 13), (13, 11), (16, 14), (14, 12),
    (5, 11), (6, 12), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1),
    (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)
]

FLIP_INDEX = [
    0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15
]

NUM_KP = 17
NUM_LIMBS = len(COCO_PERSON_SKELETON)


class COCOPoseDataset(Dataset):
    """
    COCO姿态数据集:
    返回图像张量、关键点热图和PAF张量，支持MixUp、CutMix等数据增强。
    """

    def __init__(self, root, ann_file, img_folder, img_size=(256, 192), hm_size=(64, 48),
                 sigma=2, augment=False):
        super().__init__()
        self.root = root
        ensure_coco_data(root)

        self.coco = COCO(ann_file)
        all_ids = self.coco.getImgIds(catIds=[1])  # 仅包含人的图像ID
        self.img_ids = []
        for img_id in all_ids:
            ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=[1], iscrowd=None)
            anns = self.coco.loadAnns(ann_ids)
            if any(ann['num_keypoints'] > 0 for ann in anns):
                self.img_ids.append(img_id)
        self.img_folder = img_folder
        self.img_size = img_size  # (H, W)
        self.hm_size = hm_size  # (H, W)
        self.sigma = sigma
        self.augment = augment
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=[1], iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)
        # 读取图像
        img_info = self.coco.loadImgs([img_id])[0]
        path = os.path.join(self.root, self.img_folder, img_info['file_name'])
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((img_info['height'], img_info['width'], 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_info['height'], img_info['width']
        out_h, out_w = self.img_size
        hm_h, hm_w = self.hm_size
        # 初始化heatmap和PAF标签
        num_kp = NUM_KP
        L = NUM_LIMBS
        heatmaps = np.zeros((num_kp, hm_h, hm_w), dtype=np.float32)
        paf_xy = np.zeros((2 * L, hm_h, hm_w), dtype=np.float32)
        paf_count = np.zeros((L, hm_h, hm_w), dtype=np.int32)
        # 数据增强参数
        flipped = False
        M = None
        do_mixup = False
        do_cutmix = False
        # 预处理: 缩放图像到统一尺寸
        img = cv2.resize(img, (out_w, out_h))
        if self.augment:
            # 颜色抖动 (亮度、对比度随机扰动)
            alpha = 1.0 + (random.random() * 0.4 - 0.2)  # 对比度0.8~1.2
            beta = (random.random() * 0.4 - 0.2) * 255.0  # 亮度-0.2~0.2
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
            # 随机水平翻转
            if random.random() < 0.5:
                flipped = True
                img = cv2.flip(img, 1)
            # 随机仿射: 旋转、缩放、平移
            angle = random.uniform(-30, 30)
            scale = random.uniform(0.75, 1.25)
            tx = random.uniform(-0.1, 0.1) * out_w
            ty = random.uniform(-0.1, 0.1) * out_h
            center = (out_w / 2, out_h / 2)
            M = cv2.getRotationMatrix2D(center, angle, scale)
            M[0, 2] += tx
            M[1, 2] += ty
            img = cv2.warpAffine(img, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderValue=(128, 128, 128))
            # 随机选择MixUp或CutMix
            r = random.random()
            if r < 0.2:
                do_mixup = True
            elif r < 0.4:
                do_cutmix = True
        # 处理每个人的关键点，生成GT heatmap和PAF
        for ann in anns:
            if ann['num_keypoints'] == 0:
                continue
            keypoints = np.array(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
            # 缩放关键点到输出图像尺度
            for k in range(num_kp):
                if keypoints[k, 2] > 0:
                    keypoints[k, 0] = keypoints[k, 0] * (out_w / float(orig_w))
                    keypoints[k, 1] = keypoints[k, 1] * (out_h / float(orig_h))
            # 水平翻转关键点坐标
            if flipped:
                for k in range(num_kp):
                    if keypoints[k, 2] > 0:
                        keypoints[k, 0] = out_w - 1 - keypoints[k, 0]
                keypoints = keypoints[FLIP_INDEX]
            # 仿射变换关键点
            if M is not None:
                for k in range(num_kp):
                    if keypoints[k, 2] > 0:
                        x_old, y_old = keypoints[k, 0], keypoints[k, 1]
                        new_x = M[0, 0] * x_old + M[0, 1] * y_old + M[0, 2]
                        new_y = M[1, 0] * x_old + M[1, 1] * y_old + M[1, 2]
                        if new_x < 0 or new_x >= out_w or new_y < 0 or new_y >= out_h:
                            keypoints[k, 2] = 0
                        else:
                            keypoints[k, 0] = new_x
                            keypoints[k, 1] = new_y
            # 绘制关键点高斯热图
            for k in range(num_kp):
                v = keypoints[k, 2]
                if v > 0:
                    hm_x = keypoints[k, 0] * (hm_w / float(out_w))
                    hm_y = keypoints[k, 1] * (hm_h / float(out_h))
                    ul = [int(hm_x - 3 * self.sigma), int(hm_y - 3 * self.sigma)]
                    br = [int(hm_x + 3 * self.sigma), int(hm_y + 3 * self.sigma)]
                    ul[0] = max(0, ul[0]);
                    ul[1] = max(0, ul[1])
                    br[0] = min(hm_w - 1, br[0]);
                    br[1] = min(hm_h - 1, br[1])
                    for yy in range(ul[1], br[1] + 1):
                        for xx in range(ul[0], br[0] + 1):
                            d2 = (xx - hm_x) ** 2 + (yy - hm_y) ** 2
                            if d2 <= 9 * (self.sigma ** 2):
                                gauss_val = math.exp(-d2 / (2 * (self.sigma ** 2)))
                                heatmaps[k, yy, xx] = max(heatmaps[k, yy, xx], gauss_val)
            # 绘制PAF向量场
            for c, (a, b) in enumerate(COCO_PERSON_SKELETON):
                xa, ya, va = keypoints[a]
                xb, yb, vb = keypoints[b]
                if va > 0 and vb > 0:
                    xa_hm = xa * (hm_w / float(out_w))
                    ya_hm = ya * (hm_h / float(out_h))
                    xb_hm = xb * (hm_w / float(out_w))
                    yb_hm = yb * (hm_h / float(out_h))
                    dx = xb_hm - xa_hm
                    dy = yb_hm - ya_hm
                    norm = math.hypot(dx, dy) + 1e-8
                    vx = dx / norm
                    vy = dy / norm
                    min_x = int(max(0, min(xa_hm, xb_hm) - 1))
                    max_x = int(min(hm_w - 1, max(xa_hm, xb_hm) + 1))
                    min_y = int(max(0, min(ya_hm, yb_hm) - 1))
                    max_y = int(min(hm_h - 1, max(ya_hm, yb_hm) + 1))
                    for yy in range(min_y, max_y + 1):
                        for xx in range(min_x, max_x + 1):
                            dx2 = xx - xa_hm
                            dy2 = yy - ya_hm
                            t = (dx2 * dx + dy2 * dy) / (norm ** 2)
                            if 0 <= t <= 1:
                                proj_x = xa_hm + t * dx
                                proj_y = ya_hm + t * dy
                                dist = math.hypot(xx - proj_x, yy - proj_y)
                                if dist <= 1.0:
                                    paf_xy[2 * c, yy, xx] += vx
                                    paf_xy[2 * c + 1, yy, xx] += vy
                                    paf_count[c, yy, xx] += 1
        # 平均PAF重叠区域的向量值
        for c in range(L):
            mask = paf_count[c] > 0
            if np.any(mask):
                paf_xy[2 * c, mask] /= paf_count[c, mask]
                paf_xy[2 * c + 1, mask] /= paf_count[c, mask]
        # MixUp/CutMix 数据增强
        if self.augment and (do_mixup or do_cutmix):
            # 获取第二张随机样本 (不使用增强，保证标签准确)
            j = random.randrange(len(self.img_ids))
            if j == idx:
                j = (j + 1) % len(self.img_ids)
            img_info2 = self.coco.loadImgs([self.img_ids[j]])[0]
            img2 = cv2.imread(os.path.join(self.root, self.img_folder, img_info2['file_name']))
            if img2 is None:
                img2 = np.zeros((img_info2['height'], img_info2['width'], 3), dtype=np.uint8)
            else:
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            img2 = cv2.resize(img2, (out_w, out_h))
            ann_ids2 = self.coco.getAnnIds(imgIds=[img_info2['id']], catIds=[1], iscrowd=None)
            anns2 = self.coco.loadAnns(ann_ids2)
            heatmaps2 = np.zeros_like(heatmaps)
            paf_xy2 = np.zeros_like(paf_xy)
            paf_count2 = np.zeros((L, hm_h, hm_w), dtype=np.int32)
            for ann2 in anns2:
                if ann2['num_keypoints'] == 0:
                    continue
                kps = np.array(ann2['keypoints'], dtype=np.float32).reshape(-1, 3)
                for k in range(NUM_KP):
                    if kps[k, 2] > 0:
                        kps[k, 0] = kps[k, 0] * (out_w / float(img_info2['width']))
                        kps[k, 1] = kps[k, 1] * (out_h / float(img_info2['height']))
                # 生成第二张图像的heatmap/PAF
                for k in range(NUM_KP):
                    if kps[k, 2] > 0:
                        hm_x = kps[k, 0] * (hm_w / float(out_w))
                        hm_y = kps[k, 1] * (hm_h / float(out_h))
                        ul = [int(hm_x - 3 * self.sigma), int(hm_y - 3 * self.sigma)]
                        br = [int(hm_x + 3 * self.sigma), int(hm_y + 3 * self.sigma)]
                        ul[0] = max(0, ul[0]);
                        ul[1] = max(0, ul[1])
                        br[0] = min(hm_w - 1, br[0]);
                        br[1] = min(hm_h - 1, br[1])
                        for yy in range(ul[1], br[1] + 1):
                            for xx in range(ul[0], br[0] + 1):
                                d2 = (xx - hm_x) ** 2 + (yy - hm_y) ** 2
                                if d2 <= 9 * (self.sigma ** 2):
                                    val = math.exp(-d2 / (2 * (self.sigma ** 2)))
                                    heatmaps2[k, yy, xx] = max(heatmaps2[k, yy, xx], val)
                for c, (a, b) in enumerate(COCO_PERSON_SKELETON):
                    xa, ya, va = kps[a];
                    xb, yb, vb = kps[b]
                    if va > 0 and vb > 0:
                        xa_hm = xa * (hm_w / float(out_w))
                        ya_hm = ya * (hm_h / float(out_h))
                        xb_hm = xb * (hm_w / float(out_w))
                        yb_hm = yb * (hm_h / float(out_h))
                        dx = xb_hm - xa_hm;
                        dy = yb_hm - ya_hm
                        norm = math.hypot(dx, dy) + 1e-8
                        vx = dx / norm;
                        vy = dy / norm
                        min_x = int(max(0, min(xa_hm, xb_hm) - 1))
                        max_x = int(min(hm_w - 1, max(xa_hm, xb_hm) + 1))
                        min_y = int(max(0, min(ya_hm, yb_hm) - 1))
                        max_y = int(min(hm_h - 1, max(ya_hm, yb_hm) + 1))
                        for yy in range(min_y, max_y + 1):
                            for xx in range(min_x, max_x + 1):
                                dx2 = xx - xa_hm;
                                dy2 = yy - ya_hm
                                t = (dx2 * dx + dy2 * dy) / (norm ** 2)
                                if 0 <= t <= 1:
                                    proj_x = xa_hm + t * dx;
                                    proj_y = ya_hm + t * dy
                                    dist = math.hypot(xx - proj_x, yy - proj_y)
                                    if dist <= 1.0:
                                        paf_xy2[2 * c, yy, xx] += vx
                                        paf_xy2[2 * c + 1, yy, xx] += vy
                                        paf_count2[c, yy, xx] += 1
            for c in range(L):
                mask = paf_count2[c] > 0
                if np.any(mask):
                    paf_xy2[2 * c, mask] /= paf_count2[c, mask]
                    paf_xy2[2 * c + 1, mask] /= paf_count2[c, mask]
            if do_mixup:
                lam = random.random()
                img = img.astype(np.float32) * lam + img2.astype(np.float32) * (1 - lam)
                heatmaps = heatmaps * lam + heatmaps2 * (1 - lam)
                paf_xy = paf_xy * lam + paf_xy2 * (1 - lam)
            elif do_cutmix:
                patch_w = int(out_w * random.uniform(0.3, 0.7))
                patch_h = int(out_h * random.uniform(0.3, 0.7))
                px = random.randint(0, out_w - patch_w)
                py = random.randint(0, out_h - patch_h)
                img[py:py + patch_h, px:px + patch_w] = img2[py:py + patch_h, px:px + patch_w]
                # 将对应区域的GT替换为第二张
                y1 = int(py * hm_h / out_h);
                y2 = int((py + patch_h) * hm_h / out_h)
                x1 = int(px * hm_w / out_w);
                x2 = int((px + patch_w) * hm_w / out_w)
                heatmaps[:, y1:y2, x1:x2] = heatmaps2[:, y1:y2, x1:x2]
                paf_xy[:, y1:y2, x1:x2] = paf_xy2[:, y1:y2, x1:x2]
        # 转换图像为张量并标准化
        img_tensor = self.transform(Image.fromarray(img.astype(np.uint8)))
        return img_tensor, torch.from_numpy(heatmaps), torch.from_numpy(paf_xy)


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