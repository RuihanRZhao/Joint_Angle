import os
import time
import numpy as np
from PIL import Image
import zipfile
from tqdm import tqdm
import requests

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from pycocotools.coco import COCO

from .encoder_decoder import keypoints_to_heatmaps

class COCOPoseDataset(Dataset):
    """Custom Dataset for COCO keypoint data (single-person)."""

    def __init__(self,
                 root,
                 ann_file,
                 img_dir,
                 input_size=(384, 216),
                 transform=None,
                 return_meta=False,
                 max_samples=None  # 新增：最大样本数量上限
                 ):
        """
        ann_file: path to COCO keypoints annotation JSON (e.g., person_keypoints_train2017.json)
        img_dir: directory containing the images (e.g., train2017 folder)
        max_samples: 如果指定，则只保留前 N 条 annotation，用于快速调试或小规模训练
        """
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
                self.annotations.append(ann)
        # Sort annotations by image_id for consistency
        self.annotations.sort(key=lambda a: a['image_id'])

        # 如果指定了 max_samples，则截断列表
        if max_samples is not None and isinstance(max_samples, int):
            self.annotations = self.annotations[:max_samples]

        # Flag to determine return type (for training vs evaluation)
        self.return_meta = return_meta

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
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_width - x)
        h = min(h, img_height - y)
        # Crop and resize the image to the input size
        crop_box = (int(x), int(y), int(x + w), int(y + h))
        img_crop = img.crop(crop_box)
        input_w, input_h = self.input_size
        # Preserve aspect ratio by padding if needed:
        # Compute aspect ratios
        orig_w, orig_h = img_crop.size
        target_ratio = input_w / input_h
        orig_ratio = orig_w / orig_h if orig_h != 0 else target_ratio
        if abs(orig_ratio - target_ratio) < 1e-6:
            img_resized = img_crop.resize((input_w, input_h), Image.BILINEAR)
            pad_left = pad_top = pad_right = pad_bottom = 0
        else:
            # Determine scale to fit within target while preserving aspect
            scale = min(input_w / orig_w, input_h / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            img_scaled = img_crop.resize((new_w, new_h), Image.BILINEAR)
            # Create new image with black background
            img_resized = Image.new('RGB', (input_w, input_h))
            # paste scaled image at top-left (could center, but top-left for simplicity)
            img_resized.paste(img_scaled, (0, 0))
            pad_left = pad_top = 0
            pad_right = input_w - new_w
            pad_bottom = input_h - new_h
        # Apply image transformations (to tensor and normalization)
        if self.transform:
            img_tensor = self.transform(img_resized)
        else:
            img_tensor = transforms.ToTensor()(img_resized)  # convert to [0,1] float tensor
            img_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])(img_tensor)
        # If return_meta, skip target generation and return metadata
        if self.return_meta:
            # Use the augmented (padded/clamped) bounding box as meta
            meta_bbox = np.array([x, y, w, h], dtype=np.float32)
            meta = {
                'image_id': ann['image_id'],
                'bbox': meta_bbox
            }
            return img_tensor, meta
        # Otherwise, generate target heatmaps and mask
        # Adjust keypoints coordinates to cropped & resized image space
        keypoints[:, 0] -= x
        keypoints[:, 1] -= y
        scale_x = input_w / (w if w != 0 else 1)
        scale_y = input_h / (h if h != 0 else 1)

        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y

        keypoints[:, 0] += pad_left
        keypoints[:, 1] += pad_top

        keypoints_pixel = keypoints.copy()

        keypoints[:, 0] = keypoints[:, 0] / (input_w - 1) * 2 - 1
        keypoints[:, 1] = keypoints[:, 1] / (input_h - 1) * 2 - 1

        # 生成目标热图和mask（热图大小为输入的1/4）
        out_w = int(input_w // 4)
        out_h = int(input_h // 4)
        target_heatmaps = keypoints_to_heatmaps(keypoints_pixel, input_size=(input_h, input_w), output_size=(out_h, out_w), sigma=2)
        # mask 和 tensor 构建
        mask = np.where(keypoints[:, 2] > 0, 1.0, 0.0).astype(np.float32)

        target_heatmaps_tensor = torch.from_numpy(target_heatmaps).to(dtype=torch.float32)
        mask_tensor = torch.from_numpy(mask)

        keypoints_tensor = torch.from_numpy(keypoints[:, :2]).to(dtype=torch.float32)

        return img_tensor, target_heatmaps_tensor, keypoints_tensor, mask_tensor

    def __len__(self):
        return len(self.annotations)


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
        "train2017.zip":   "http://images.cocodataset.org/zips/train2017.zip",
        "val2017.zip":     "http://images.cocodataset.org/zips/val2017.zip",
        "annotations.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    }

    for fname, url in urls.items():
        zip_path = os.path.join(root, fname)
        target_folder = {
            "train2017.zip":   os.path.join(root, "train2017"),
            "val2017.zip":     os.path.join(root, "val2017"),
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
                # 获取所有文件的列表
                file_list = z.namelist()

                # 使用tqdm显示解压进度
                with tqdm(total=len(file_list), desc="Unzipping", unit="file") as pbar:
                    for file in file_list:
                        z.extract(file, root)
                        pbar.update(1)

        except zipfile.BadZipFile as e:
            raise RuntimeError(f"解压 {zip_path} 失败：{e}")
