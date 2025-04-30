import os
import subprocess
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
from torchvision.tv_tensors import Image as TVImage, Mask, BoundingBoxes
from torchvision.datasets import CocoDetection


class COCODownloader:
    """COCO数据集自动下载与验证工具（使用wget+unzip）"""

    # 官方下载链接
    IMAGE_ZIPS = {
        'train2017': 'http://images.cocodataset.org/zips/train2017.zip',
        'val2017': 'http://images.cocodataset.org/zips/val2017.zip'
    }
    ANNO_ZIP = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    REQUIRED_FILES = [
        'images/train2017',
        'images/val2017',
        'annotations/person_keypoints_train2017.json',
        'annotations/person_keypoints_val2017.json'
    ]

    def __init__(self, root_dir: str):
        self.root = Path(root_dir).resolve()

    def check_dataset(self) -> bool:
        return all((self.root / f).exists() for f in self.REQUIRED_FILES)

    def download(self):
        if self.check_dataset():
            print("COCO数据集已存在且完整。")
            return

        # 创建目录
        (self.root / "images").mkdir(parents=True, exist_ok=True)
        (self.root / "annotations").mkdir(parents=True, exist_ok=True)

        # 下载和解压图片集
        for split, url in self.IMAGE_ZIPS.items():
            zip_path = self.root / "images" / f"{split}.zip"
            out_dir = self.root / "images"
            if not (out_dir / split).exists():
                print(f"下载 {split} ...")
                self._run_cmd(f'wget {url} -O {zip_path}')
                print(f"解压 {zip_path} ...")
                self._run_cmd(f'unzip -q {zip_path} -d {out_dir}')
                os.remove(zip_path)

        # 下载和解压标注
        anno_zip_path = self.root / "annotations" / "annotations_trainval2017.zip"
        if not (self.root / "annotations" / "person_keypoints_train2017.json").exists():
            print("下载 annotations ...")
            self._run_cmd(f'wget {self.ANNO_ZIP} -O {anno_zip_path}')
            print(f"解压 {anno_zip_path} ...")
            self._run_cmd(f'unzip -q {anno_zip_path} -d {self.root}/annotations')
            os.remove(anno_zip_path)

        if not self.check_dataset():
            missing = [f for f in self.REQUIRED_FILES if not (self.root / f).exists()]
            raise RuntimeError(f"下载完成后仍缺失: {missing}")

    def _run_cmd(self, cmd: str):
        print(f"执行: {cmd}")
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            raise RuntimeError(f"命令失败: {cmd}")


class CocoMultiTask(CocoDetection):
    """支持自动下载的COCO多任务数据集"""

    def __init__(
            self,
            root: str,
            split: str = 'train',
            transform: Optional[T.Compose] = None,
            img_size: Tuple[int, int] = (512, 512),
            auto_download: bool = True
    ):
        # 初始化路径
        self.root_path = Path(root)
        self.split = split

        # 自动下载数据集
        if auto_download:
            self._ensure_dataset_ready()

        # 设置路径
        self.ann_file = self.root_path / f"annotations/person_keypoints_{self.split}2017.json"
        self.img_dir = self.root_path / "images" / f"{self.split}2017"

        super().__init__(
            root=self.img_dir,
            annFile=self.ann_file,
            transform=transform,
            transforms=None
        )

        # 过滤有效标注
        self.ids = self._filter_images()
        self._keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]

    def _ensure_dataset_ready(self):
        """确保数据集已下载"""
        downloader = COCODownloader(self.root_path)
        if not downloader.check_dataset():
            print(f"开始下载COCO {self.split}数据集...")
            downloader.download()

        if not self.ann_file.exists():
            raise FileNotFoundError(f"标注文件缺失: {self.ann_file}")
        if not self.img_dir.exists():
            raise NotADirectoryError(f"图像目录缺失: {self.img_dir}")


    def _filter_images(self):
        """过滤包含有效标注的图像"""
        valid_ids = []
        for img_id in self.coco.getImgIds():
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            anns = self.coco.loadAnns(ann_ids)
            if self._has_valid_annotation(anns):
                valid_ids.append(img_id)
        return valid_ids

    def _has_valid_annotation(self, anns):
        """验证至少包含一个有效人体实例"""
        for ann in anns:
            if ann.get('num_keypoints', 0) >= 5 or ann.get('iscrowd', 0) == 0:
                return True
        return False

    def __getitem__(self, index):
        img, targets = super().__getitem__(index)

        # 解析标注
        mask = self._parse_segmentation(targets, img.size[::-1])
        keypoints, visibilities = self._parse_keypoints(targets)

        # 转换格式
        img_tensor = TVImage(img)
        mask_tensor = Mask(mask)

        # 应用变换
        if self.transform is not None:
            img_tensor, mask_tensor = self.transform(img_tensor, mask_tensor)

        return img_tensor, mask_tensor, keypoints, visibilities

    def _parse_segmentation(self, anns, img_size):
        """生成分割掩码"""
        mask = np.zeros(img_size, dtype=np.uint8)
        for ann in anns:
            if ann['iscrowd']:
                rle = coco_mask.frPyObjects(ann['segmentation'], *img_size)
                m = coco_mask.decode(rle)
            else:
                m = self.coco.annToMask(ann)
            mask = np.maximum(mask, m)
        return mask

    def _parse_keypoints(self, anns):
        """解析关键点数据"""
        keypoints = []
        visibilities = []
        for ann in anns:
            if 'keypoints' in ann:
                kps = np.array(ann['keypoints']).reshape(-1, 3)
                valid = kps[:, 2] > 0  # 可见性过滤
                keypoints.append(kps[valid, :2])
                visibilities.append(valid.astype(np.float32))
        return np.concatenate(keypoints), np.concatenate(visibilities)


def get_transform(train: bool = True) -> T.Compose:
    """数据增强管道"""
    transforms = []
    if train:
        transforms += [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(
                degrees=(-5, 5),
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
        ]
    transforms += [
        T.Resize(size=(512, 512), antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
    return T.Compose(transforms)


def collate_fn(batch):
    """自定义批处理函数"""
    images, masks, keypoints, visibilities = zip(*batch)
    return (
        torch.stack(images),
        torch.stack(masks),
        [kp for kp in keypoints],
        [vis for vis in visibilities]
    )


# 使用示例
if __name__ == "__main__":
    # 初始化数据集（自动下载）
    dataset = CocoMultiTask(
        root="./coco_data",
        split='val',
        transform=get_transform(train=False),
        auto_download=True
    )

    print(f"成功加载数据集，包含 {len(dataset)} 个样本")
