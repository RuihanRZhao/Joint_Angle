import os

import json
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from torchvision import transforms

from pycocotools.coco import COCO





import zipfile
from typing import Tuple, List
from tqdm import tqdm
import requests

import time
import math

class COCOPoseDataset(Dataset):
    """Custom Dataset for COCO keypoint data (single-person)."""

    def __init__(self, root, ann_file, img_dir, input_size=(384, 216), transform=None):
        """
        ann_file: path to COCO keypoints annotation JSON (e.g., person_keypoints_train2017.json)
        img_dir: directory containing the images (e.g., train2017 folder)
        input_size: tuple (width, height) for resized input images
        transform: optional torchvision transforms to apply to the image (after cropping/resizing)
        """
        ensure_coco_data(root, retries=2, backoff_factor=1.0)

        ann_path = os.path.join(root, ann_file)
        if not os.path.isfile(ann_path):
            raise RuntimeError(f"找不到注解文件：{ann_path}")
        self.coco = COCO(ann_path)

        self.img_dir = os.path.join(root, img_dir)
        if not os.path.isdir(self.img_dir):
            raise RuntimeError(f"找不到图像目录：{self.img_dir}")


        self.input_size = tuple(input_size)
        self.transform = transform
        # Gather all person annotations with keypoints
        self.annotations = []
        ann_ids = self.coco.getAnnIds(catIds=[1])  # category 1 is person in COCO
        for ann in self.coco.loadAnns(ann_ids):
            if ann.get('num_keypoints', 0) > 0:
                # Only consider annotations that have at least one keypoint
                self.annotations.append(ann)



    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Load annotation and corresponding image
        ann = self.annotations[idx]
        image_info = self.coco.loadImgs(ann['image_id'])[0]
        img_path = os.path.join(self.img_dir, image_info['file_name'])
        # Open image
        img = Image.open(img_path).convert('RGB')
        # Keypoints and bounding box from annotation
        keypoints = np.array(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
        bbox = ann.get('bbox', None)  # [x, y, w, h]
        if bbox is None:
            # If no bbox provided, compute from keypoints as min/max (not typical for COCO, since bbox is provided)
            visible_pts = keypoints[keypoints[:, 2] > 0]
            if visible_pts.size == 0:
                # No visible keypoints, return zeros (this case shouldn't happen given num_keypoints > 0)
                visible_pts = keypoints[:, :2]
            x_min, y_min = visible_pts[:, 0].min(), visible_pts[:, 1].min()
            x_max, y_max = visible_pts[:, 0].max(), visible_pts[:, 1].max()
            bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]
        x, y, w, h = bbox
        # Optionally expand the bounding box slightly for more context
        pad = 0.15  # 15% padding
        x_c, y_c = x + w / 2.0, y + h / 2.0
        w = w * (1 + pad)
        h = h * (1 + pad)
        x = x_c - w / 2.0
        y = y_c - h / 2.0
        # Clamp the coordinates to be within image bounds
        img_width, img_height = img.size
        x = max(0, x);
        y = max(0, y)
        w = min(w, img_width - x);
        h = min(h, img_height - y)
        # Crop and resize the image to the input size
        crop_box = (int(x), int(y), int(x + w), int(y + h))
        img_crop = img.crop(crop_box)
        input_w, input_h = self.input_size
        img_resized = img_crop.resize((input_w, input_h), Image.BILINEAR)
        # Adjust keypoints to the cropped+resized image coordinates
        keypoints[:, 0] -= x
        keypoints[:, 1] -= y
        scale_x = input_w / (w if w != 0 else 1)  # avoid div by zero
        scale_y = input_h / (h if h != 0 else 1)
        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y
        # Apply image transformations (to tensor and normalization, plus any augmentation if provided)
        if self.transform:
            img_tensor = self.transform(img_resized)
        else:
            img_tensor = transforms.ToTensor()(img_resized)  # convert to [0,1] float tensor
            img_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])(img_tensor)
        # Generate target heatmaps for keypoints
        num_joints = keypoints.shape[0]
        out_w, out_h = input_w // 4, input_h // 4  # output heatmap size (assuming 1/4 resolution)
        target_heatmaps = np.zeros((num_joints, out_h, out_w), dtype=np.float32)
        sigma = 2.0
        tmp_size = sigma * 3
        # For each joint, draw a Gaussian on the heatmap
        for j in range(num_joints):
            x_j, y_j, v_j = keypoints[j]
            if v_j < 1:  # v=0 (not labeled) -> no target
                continue
            # Compute coordinate in heatmap space
            x_hm = x_j * (out_w / input_w)
            y_hm = y_j * (out_h / input_h)
            jx = int(np.round(x_hm))
            jy = int(np.round(y_hm))
            if jx < 0 or jx >= out_w or jy < 0 or jy >= out_h:
                # Skip if joint is outside the bounds after transform (shouldn't happen often)
                continue
            # Determine Gaussian bounds on the heatmap
            x0 = max(0, jx - int(tmp_size))
            y0 = max(0, jy - int(tmp_size))
            x1 = min(out_w, jx + int(tmp_size) + 1)
            y1 = min(out_h, jy + int(tmp_size) + 1)
            # Create mesh grid for the Gaussian patch
            gx = np.arange(x0, x1)
            gy = np.arange(y0, y1)[:, np.newaxis]
            # Gaussian distribution centered at (jx, jy)
            gaussian = np.exp(-((gx - jx) ** 2 + (gy - jy) ** 2) / (2 * sigma ** 2))
            # Place the Gaussian patch in the heatmap (using maximum in case of overlap)
            target_heatmaps[j, y0:y1, x0:x1] = np.maximum(target_heatmaps[j, y0:y1, x0:x1], gaussian)
        # Convert to torch tensor
        target_heatmaps = torch.from_numpy(target_heatmaps)
        return img_tensor, target_heatmaps

def ensure_coco_data(root, retries: int = 3, backoff_factor: float = 2.0):
    """
    确保 root 下存在 COCO 2017 的 train/val 图像和注解。
    如果缺失，则从官方 URL 下载并解压。
    支持重试机制和友好提示。

    Args:
        root (str): 数据根目录，比如 "data/coco"
        retries (int): 最多重试次数
        backoff_factor (float): 每次重试等待时间的增长因子
    Raises:
        RuntimeError: 下载或解压多次失败后抛出
    """
    os.makedirs(root, exist_ok=True)

    # COCO 官方下载链接
    urls = {
        "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
        "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
        "annotations.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    }

    for fname, url in urls.items():
        zip_path = os.path.join(root, fname)
        target_folder = {
            "train2017.zip": os.path.join(root, "train2017"),
            "val2017.zip": os.path.join(root, "val2017"),
            "annotations.zip": os.path.join(root, "annotations"),
        }[fname]

        # 如果文件夹已经存在，则跳过
        if os.path.isdir(target_folder):
            continue

        # 否则，需要下载并解压
        for attempt in range(1, retries + 1):
            try:
                print(f"[COCO] 第 {attempt} 次尝试下载 {fname} ...")
                resp = requests.get(url, stream=True, timeout=10)
                resp.raise_for_status()

                # 写入本地 zip
                total = int(resp.headers.get('content-length', 0))
                with open(zip_path, 'wb') as f, tqdm(
                        desc=f"Downloading {fname}",
                        total=total,
                        unit='B',
                        unit_scale=True
                ) as pbar:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

                # 成功下载，跳出重试循环
                break

            except requests.exceptions.RequestException as e:
                print(f"[COCO] 下载失败（{e}）")
                if attempt < retries:
                    wait = backoff_factor ** (attempt - 1)
                    print(f"[COCO] {wait:.1f}s 后重试…")
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"多次下载 {fname} 失败，请检查网络或手动下载安装：{url}")

        # 解压 zip
        print(f"[COCO] 解压 {zip_path} → {target_folder}")
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(root)
        except zipfile.BadZipFile as e:
            raise RuntimeError(f"解压 {zip_path} 失败：{e}")

        # 删除 zip 文件，节省空间
        try:
            os.remove(zip_path)
        except OSError:
            pass