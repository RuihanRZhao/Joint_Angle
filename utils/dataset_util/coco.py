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

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2

class COCOPoseDataset(Dataset):
    def __init__(self,
                 root,
                 ann_file,
                 img_dir,
                 input_size=(384, 384),
                 return_meta=False,
                 max_samples=0,
                 bins=4,
                 downsample=4):
        self.coco = COCO(os.path.join(root, ann_file))
        self.img_dir = os.path.join(root, img_dir)
        self.input_w, self.input_h = input_size
        self.return_meta = return_meta
        self.bins = bins
        self.downsample = downsample

        self.out_w = self.input_w // downsample
        self.out_h = self.input_h // downsample
        self.x_classes = self.out_w * bins
        self.y_classes = self.out_h * bins

        ann_ids = self.coco.getAnnIds(catIds=[1])
        anns = [self.coco.loadAnns(i)[0] for i in ann_ids]

        valid_annotations = []
        for a in anns:
            kps = np.array(a.get('keypoints', []), dtype=np.float32).reshape(-1, 3)
            if (kps[:, 2] > 0).sum() >= 10:
                valid_annotations.append(a)

        self.annotations = valid_annotations[:max_samples] if max_samples > 0 else valid_annotations

        # Albumentations transform pipeline
        self.transform = A.Compose([
            A.LongestMaxSize(max_size=max(self.input_w, self.input_h)),
            A.PadIfNeeded(min_height=self.input_h, min_width=self.input_w, border_mode=0),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=30, p=0.4),
            A.RandomScale(scale_limit=0.25, p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.CoarseDropout(max_holes=2, max_height=32, max_width=32, p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_info = self.coco.loadImgs(ann['image_id'])[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = np.array(Image.open(img_path).convert('RGB'))

        kps = np.array(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
        keypoints_xy = [(x, y) for x, y, v in kps]
        visibility = kps[:, 2]

        # Apply transform
        augmented = self.transform(image=img, keypoints=keypoints_xy)
        img_tensor = augmented['image']
        keypoints = np.array(augmented['keypoints'])

        # Create SimCC labels
        K = key




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
