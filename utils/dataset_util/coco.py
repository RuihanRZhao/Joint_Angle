import os
import numpy as np
from PIL import Image
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
