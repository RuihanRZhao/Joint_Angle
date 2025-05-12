import os
import numpy as np
import time
from PIL import Image
import zipfile
from tqdm import tqdm
import requests
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pycocotools.coco import COCO
import math


class COCOPoseDataset(Dataset):
    def __init__(self,
                 root,
                 ann_file,
                 img_dir,
                 input_size=(384, 384),
                 transform=None,
                 return_meta=False,
                 max_samples=0,
                 bins=4,
                 downsample=4,
                 use_soft_label=True,
                 soft_sigma=1.5,
                 min_keypoints=5):
        self.coco = COCO(os.path.join(root, ann_file))
        self.img_dir = os.path.join(root, img_dir)
        self.input_w, self.input_h = input_size
        self.return_meta = return_meta
        self.transform = transform
        self.bins = bins
        self.downsample = downsample
        self.out_w = self.input_w // downsample
        self.out_h = self.input_h // downsample
        self.x_classes = self.out_w * bins
        self.y_classes = self.out_h * bins
        self.use_soft_label = use_soft_label
        self.soft_sigma = soft_sigma
        self.min_keypoints = min_keypoints

        ann_ids = self.coco.getAnnIds(catIds=[1])
        anns = [self.coco.loadAnns(i)[0] for i in ann_ids]

        valid_annotations = []
        for a in anns:
            kps = np.array(a.get('keypoints', []), dtype=np.float32).reshape(-1, 3)
            visible_count = int((kps[:, 2] > 0).sum())
            if visible_count >= self.min_keypoints:
                valid_annotations.append(a)

        self.annotations = sorted(valid_annotations, key=lambda x: x['image_id'])
        if max_samples > 0:
            self.annotations = self.annotations[:max_samples]

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def _gaussian_label(self, size, center_idx, sigma):
        x = np.arange(0, size)
        gauss = np.exp(-0.5 * ((x - center_idx) / sigma) ** 2)
        return gauss / (gauss.sum() + 1e-6)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_info = self.coco.loadImgs(ann['image_id'])[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')

        kps = np.array(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
        bbox = ann.get('bbox', [0, 0, img.width, img.height])
        x, y, w, h = bbox

        # Padding bbox
        pad = 0.15
        xc, yc = x + w / 2, y + h / 2
        w *= (1 + pad)
        h *= (1 + pad)
        x = max(0, xc - w / 2)
        y = max(0, yc - h / 2)
        w = min(w, img.width - x)
        h = min(h, img.height - y)

        crop = img.crop((int(x), int(y), int(x + w), int(y + h)))
        crop = crop.resize((self.input_w, self.input_h), Image.BILINEAR)
        img_tensor = self.transform(crop)

        if self.return_meta:
            return img_tensor, {
                'image_id': ann['image_id'],
                'bbox': np.array([x, y, w, h], dtype=np.float32)
            }

        # 坐标变换
        kps[:, 0] = (kps[:, 0] - x) * (self.input_w / w)
        kps[:, 1] = (kps[:, 1] - y) * (self.input_h / h)

        target_x = np.zeros((kps.shape[0], self.x_classes), dtype=np.float32)
        target_y = np.zeros((kps.shape[0], self.y_classes), dtype=np.float32)
        mask = np.zeros((kps.shape[0],), dtype=np.float32)

        for j, (kx, ky, v) in enumerate(kps):
            if v <= 0:
                continue
            mask[j] = 1.0
            x_norm = np.clip(kx / self.downsample, 0, self.out_w - 1e-3)
            y_norm = np.clip(ky / self.downsample, 0, self.out_h - 1e-3)
            idx_x = int(np.floor(x_norm * self.bins))
            idx_y = int(np.floor(y_norm * self.bins))

            if self.use_soft_label:
                target_x[j] = self._gaussian_label(self.x_classes, idx_x, sigma=self.soft_sigma)
                target_y[j] = self._gaussian_label(self.y_classes, idx_y, sigma=self.soft_sigma)
            else:
                target_x[j, idx_x] = 1.0
                target_y[j, idx_y] = 1.0

        return (
            img_tensor,
            torch.from_numpy(target_x),
            torch.from_numpy(target_y),
            torch.from_numpy(mask)
        )


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