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

class COCOPoseDataset(Dataset):
    """
    COCO 单人关键点数据集（SimCC 坐标分类版本）
    输出:
      - image_tensor:      [3, H, W], float32
      - target_x:          [K, out_W * bins], float32, one-hot
      - target_y:          [K, out_H * bins], float32, one-hot
      - mask:              [K], float32, 1 表示该关节有效 (v>0)，0 表示无标注 (v==0)
    或者若 return_meta=True:
      - image_tensor, { 'image_id': id, 'bbox': [x,y,w,h] }
    """
    def __init__(self,
                 root,
                 ann_file,
                 img_dir,
                 input_size=(384, 216),
                 transform=None,
                 return_meta=False,
                 max_samples=0,
                 bins=4,                # 每像素细分子区数量
                 downsample=4           # 下采样倍数，与模型解码一致
                 ):
        self.coco = COCO(os.path.join(root, ann_file))
        self.img_dir = os.path.join(root, img_dir)
        self.input_w, self.input_h = input_size[0], input_size[1]
        self.return_meta = return_meta
        self.transform = transform
        self.bins = bins
        self.downsample = downsample
        self.min_keypoints = 10

        # 输出分类维度
        self.out_w = self.input_w // self.downsample
        self.out_h = self.input_h // self.downsample
        self.x_classes = self.out_w * self.bins
        self.y_classes = self.out_h * self.bins

        # 加载标注
        ann_ids = self.coco.getAnnIds(catIds=[1])
        anns = [self.coco.loadAnns(i)[0] for i in ann_ids]

        # 过滤无关键点样本
        valid_annotations = []
        for a in anns:
            kps = np.array(a.get('keypoints', []), dtype=np.float32).reshape(-1, 3)
            visible_count = int((kps[:, 2] > 0).sum())
            if visible_count >= self.min_keypoints:
                valid_annotations.append(a)

        self.annotations = sorted(valid_annotations, key=lambda x: x['image_id'])
        if max_samples != 0:
            self.annotations = self.annotations[:max_samples]

        # 默认变换
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),  # [0,255] -> [0,1]
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_info = self.coco.loadImgs(ann['image_id'])[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')

        # 原始关键点 & bbox
        kps = np.array(ann['keypoints'], dtype=np.float32).reshape(-1,3)  # [K,3]
        bbox = ann.get('bbox', None)
        if bbox is None:
            visible = kps[:,2]>0
            pts = kps[visible,:2] if visible.any() else kps[:,:2]
            x0,y0 = pts[:,0].min(), pts[:,1].min()
            x1,y1 = pts[:,0].max(), pts[:,1].max()
            bbox = [x0, y0, x1-x0, y1-y0]
        x, y, w, h = bbox

        # 扩展 bbox 15%
        pad = 0.15
        xc, yc = x + w/2, y + h/2
        w *= (1+pad); h *= (1+pad)
        x = max(0, xc - w/2); y = max(0, yc - h/2)
        w = min(w, img.width - x); h = min(h, img.height - y)

        # 裁剪 & 缩放（保持比例 + 左上填充）
        crop = img.crop((int(x),int(y),int(x+w),int(y+h)))
        orig_w, orig_h = crop.size
        target_ratio = self.input_w / self.input_h
        orig_ratio   = orig_w / orig_h if orig_h>0 else target_ratio

        if abs(orig_ratio - target_ratio) < 1e-6:
            resized = crop.resize((self.input_w, self.input_h), Image.BILINEAR)
            pad_left = pad_top = pad_right = pad_bottom = 0
        else:
            scale = min(self.input_w/orig_w, self.input_h/orig_h)
            nw, nh = int(orig_w*scale), int(orig_h*scale)
            scaled = crop.resize((nw, nh), Image.BILINEAR)
            resized = Image.new('RGB', (self.input_w, self.input_h))
            resized.paste(scaled, (0,0))
            pad_left = pad_top = 0
            pad_right = self.input_w - nw
            pad_bottom= self.input_h - nh

        # 图像变换
        img_tensor = self.transform(resized)

        # 元信息模式
        if self.return_meta:
            meta = {'image_id': ann['image_id'],
                    'bbox': np.array([x,y,w,h], dtype=np.float32)}
            return img_tensor, meta

        # 调整关键点到裁剪后尺寸
        kps_px = kps.copy()
        kps_px[:,0] -= x; kps_px[:,1] -= y
        # 计算缩放比例
        if pad_right or pad_bottom:
            sx = (self.input_w - pad_right) / (w if w>0 else 1)
            sy = (self.input_h - pad_bottom)/ (h if h>0 else 1)
        else:
            sx = self.input_w / (w if w>0 else 1)
            sy = self.input_h / (h if h>0 else 1)
        kps_px[:,0] = kps_px[:,0]*sx + pad_left
        kps_px[:,1] = kps_px[:,1]*sy + pad_top

        # SimCC 目标张量
        K = kps_px.shape[0]
        target_x = np.zeros((K, self.x_classes), dtype=np.float32)
        target_y = np.zeros((K, self.y_classes), dtype=np.float32)
        mask     = np.zeros((K,), dtype=np.float32)

        for j in range(K):
            vx = kps_px[j,0]
            vy = kps_px[j,1]
            v  = kps_px[j,2]
            # 只监督 v>0 的点 (COCO v=0 无标注)
            if v > 0:
                mask[j] = 1.0
                # 归一化到 [0, out_w), [0, out_h)
                x_s = np.clip(vx / self.downsample, 0, self.out_w - 1e-3)
                y_s = np.clip(vy / self.downsample, 0, self.out_h - 1e-3)
                # 分类索引
                ix = int(np.floor(x_s * self.bins))
                iy = int(np.floor(y_s * self.bins))
                cls_x = j * 0  # placeholder, j used in axis, so:
                idx_x = int(np.clip(x_s * self.bins, 0, self.x_classes - 1))
                idx_y = int(np.clip(y_s * self.bins, 0, self.y_classes - 1))
                target_x[j, idx_x] = 1.0
                target_y[j, idx_y] = 1.0

        # 转为 Tensor
        target_x = torch.from_numpy(target_x)  # [K, x_classes]
        target_y = torch.from_numpy(target_y)  # [K, y_classes]
        mask     = torch.from_numpy(mask)      # [K]

        return img_tensor, target_x, target_y, mask



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
