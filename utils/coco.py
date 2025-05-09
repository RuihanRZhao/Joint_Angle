import math
import os
import random
import numpy as np
import torch
from pycocotools.coco import COCO
from PIL import Image
import torchvision.transforms as T
import cv2
import zipfile
from torch.utils.data import Dataset
from typing import Tuple, List
from tqdm import tqdm
import requests

import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed


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


class COCOPoseDataset(torch.utils.data.Dataset):
    """COCO多人姿态数据集。返回图像tensor及对应的关键点热图和PAF，以及人体数量。"""
    def __init__(self, root, ann_file, img_folder, img_size=(256, 256), hm_size=(64, 64), sigma=2):
        super().__init__()
        # 确保COCO数据存在
        ensure_coco_data(root)
        self.coco = COCO(ann_file)
        all_img_ids = self.coco.getImgIds(catIds=[1])  # 类别1为人类
        self.img_ids = []
        self.person_counts = []
        for img_id in all_img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=[1], iscrowd=None)
            anns = self.coco.loadAnns(ann_ids)
            # 统计该图像中有关键点标注的人数
            persons = [ann for ann in anns if ann['num_keypoints'] > 0]
            if len(persons) > 0:
                self.img_ids.append(img_id)
                self.person_counts.append(len(persons))
        self.root = root
        self.img_folder = img_folder
        self.img_size = img_size  # 网络输入图像大小
        self.hm_size = hm_size    # 输出热图/PAF大小
        self.sigma = sigma        # 高斯热图标准差
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        assert len(self.img_ids) == len(self.person_counts)
    def __len__(self):
        return len(self.img_ids)
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        # 获取该图对应的所有人体标注
        ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=[1], iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)
        # 加载原始图像并调整大小
        img_info = self.coco.loadImgs([img_id])[0]
        img_path = os.path.join(self.root, self.img_folder, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img_info['width'], img_info['height']
        # 调整图像为指定输入大小
        img = img.resize((self.img_size[1], self.img_size[0]))
        # 初始化热图和PAF张量
        num_kp = 17
        L = len(COCO_PERSON_SKELETON)
        hm = np.zeros((num_kp, self.hm_size[0], self.hm_size[1]), dtype=np.float32)
        paf_x = np.zeros((L, self.hm_size[0], self.hm_size[1]), dtype=np.float32)
        paf_y = np.zeros((L, self.hm_size[0], self.hm_size[1]), dtype=np.float32)
        count = np.zeros((L, self.hm_size[0], self.hm_size[1]), dtype=np.int32)
        # 遍历该图像中的每个人体注释
        for ann in anns:
            if ann['num_keypoints'] == 0:
                continue  # 跳过无关键点标注的人体
            keypoints = np.array(ann['keypoints']).reshape(-1, 3)
            # 绘制关键点高斯热图
            for k, (x, y, v) in enumerate(keypoints):
                if v > 0:  # v=1或2表示该关键点有标注
                    # 将原图坐标映射到热图尺寸
                    hm_x = x * (self.hm_size[1] / float(orig_w))
                    hm_y = y * (self.hm_size[0] / float(orig_h))
                    ul = [int(hm_x - 3*self.sigma), int(hm_y - 3*self.sigma)]
                    br = [int(hm_x + 3*self.sigma), int(hm_y + 3*self.sigma)]
                    ul[0] = max(0, ul[0]); ul[1] = max(0, ul[1])
                    br[0] = min(self.hm_size[1]-1, br[0]); br[1] = min(self.hm_size[0]-1, br[1])
                    for yy in range(ul[1], br[1]+1):
                        for xx in range(ul[0], br[0]+1):
                            d2 = (xx - hm_x) ** 2 + (yy - hm_y) ** 2
                            if d2 <= 9 * (self.sigma ** 2):
                                gaussian_val = np.exp(-d2 / (2 * (self.sigma ** 2)))
                                hm[k, yy, xx] = max(hm[k, yy, xx], gaussian_val)
            # 绘制骨骼连接（PAF向量场）
            for c, (a, b) in enumerate(COCO_PERSON_SKELETON):
                xa, ya, va = keypoints[a]
                xb, yb, vb = keypoints[b]
                if va > 0 and vb > 0:
                    # 将原图坐标映射到热图尺寸
                    xa_hm = xa * (self.hm_size[1] / float(orig_w))
                    ya_hm = ya * (self.hm_size[0] / float(orig_h))
                    xb_hm = xb * (self.hm_size[1] / float(orig_w))
                    yb_hm = yb * (self.hm_size[0] / float(orig_h))
                    dx = xb_hm - xa_hm
                    dy = yb_hm - ya_hm
                    norm = np.sqrt(dx**2 + dy**2) + 1e-8
                    vx = dx / norm
                    vy = dy / norm
                    # 计算连接线段覆盖的范围框
                    min_x = int(max(0, min(xa_hm, xb_hm) - 1))
                    max_x = int(min(self.hm_size[1] - 1, max(xa_hm, xb_hm) + 1))
                    min_y = int(max(0, min(ya_hm, yb_hm) - 1))
                    max_y = int(min(self.hm_size[0] - 1, max(ya_hm, yb_hm) + 1))
                    # 在骨骼连线范围内计算每个像素与连线的距离，填充PAF向量
                    for yy in range(min_y, max_y + 1):
                        for xx in range(min_x, max_x + 1):
                            # 计算当前像素到连线的投影距离
                            u = ((xx - xa_hm) * dx + (yy - ya_hm) * dy) / (norm**2)
                            if u < 0 or u > 1:
                                continue
                            proj_x = xa_hm + u * dx
                            proj_y = ya_hm + u * dy
                            dist = np.sqrt((xx - proj_x)**2 + (yy - proj_y)**2)
                            if dist <= 1:  # 设定阈值距离1像素
                                paf_x[c, yy, xx] += vx
                                paf_y[c, yy, xx] += vy
                                count[c, yy, xx] += 1
        # 对重叠区域PAF取平均
        for c in range(len(COCO_PERSON_SKELETON)):
            mask = count[c] > 0
            if np.any(mask):
                paf_x[c, mask] /= count[c, mask]
                paf_y[c, mask] /= count[c, mask]
        # 合并PAF的x和y两个通道
        paf = np.zeros((2 * len(COCO_PERSON_SKELETON), self.hm_size[0], self.hm_size[1]), dtype=np.float32)
        for c in range(len(COCO_PERSON_SKELETON)):
            paf[2*c]   = paf_x[c]
            paf[2*c+1] = paf_y[c]
        # 图像转换为Tensor并归一化
        img_tensor = self.transform(img)
        # 获取该样本人体数量
        n_person = self.person_counts[idx]
        return img_tensor, torch.from_numpy(hm), torch.from_numpy(paf), int(n_person)


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