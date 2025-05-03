import os
import json
import requests
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import numpy as np
from multiprocessing.pool import ThreadPool
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


def prepare_coco_dataset(root_dir, split='train', max_samples=None, force_reload=False, num_workers=8):
    """
    高效加载 COCO-2017 数据集：支持 'train'/'val'/'validation'，仅下载缺失文件，缓存索引，并显示进度。

    参数:
        root_dir (str): 数据根目录
        split (str): 'train' 或 'val'/'validation'
        max_samples (int|None): 最多加载样本数
        force_reload (bool): 强制重建本地索引和重新下载缺失文件
        num_workers (int): 并行下载线程数

    返回:
        List[dict]: 每项包含 image_path, mask(np.uint8), keypoints(np.float32), visibility(np.int)
    """
    # 归一化 split 名称
    norm = split.lower()
    if norm in ('validation', 'val'):
        ann_split = 'val'
    elif norm == 'train':
        ann_split = 'train'
    else:
        raise ValueError(f"Unsupported split '{split}', use 'train' or 'val'/'validation'.")

    # 创建必要目录
    ann_dir = os.path.join(root_dir, 'annotations')
    images_dir = os.path.join(root_dir, f'images/{ann_split}2017')
    os.makedirs(ann_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # annotation 文件 URL 和本地路径
    ann_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    ann_zip = os.path.join(ann_dir, 'annotations_trainval2017.zip')
    ann_file = os.path.join(ann_dir, f'instances_{ann_split}2017.json')

    # 下载并解压注释文件
    if not os.path.exists(ann_file) or force_reload:
        if not os.path.exists(ann_zip) or force_reload:
            response = requests.get(ann_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            with open(ann_zip, 'wb') as f:
                for chunk in tqdm(response.iter_content(1024 * 1024), total=max(1, total_size // (1024*1024)), desc='Downloading annotations', unit='MB'):
                    f.write(chunk)
        import zipfile
        with zipfile.ZipFile(ann_zip, 'r') as z:
            member = f'annotations/instances_{ann_split}2017.json'
            z.extract(member, ann_dir)
        extracted = os.path.join(ann_dir, member)
        if os.path.exists(extracted):
            os.replace(extracted, ann_file)
        nested = os.path.join(ann_dir, 'annotations')
        if os.path.isdir(nested):
            try: os.rmdir(nested)
            except OSError: pass

    # 加载 COCO annotation
    coco = COCO(ann_file)
    cat_ids = coco.getCatIds(['person'])
    img_ids = coco.getImgIds(catIds=cat_ids)
    if max_samples is not None:
        img_ids = img_ids[:max_samples]

    # 缓存或加载图片元数据索引
    index_path = os.path.join(root_dir, f'coco_index_{ann_split}.json')
    if os.path.exists(index_path) and not force_reload:
        with open(index_path, 'r') as f:
            img_info = json.load(f)
    else:
        img_info = {str(i): coco.loadImgs(i)[0] for i in img_ids}
        with open(index_path, 'w') as f:
            json.dump(img_info, f)

    # 下载缺失或损坏的图片，显示进度
    def download(img):
        url = img['coco_url']
        dst = os.path.join(images_dir, img['file_name'])
        r = requests.get(url, stream=True)
        with open(dst, 'wb') as f:
            for chunk in r.iter_content(1024 * 512):
                f.write(chunk)

    infos = list(img_info.values())
    to_download = [info for info in infos
                   if not os.path.exists(os.path.join(images_dir, info['file_name']))
                   or os.path.getsize(os.path.join(images_dir, info['file_name'])) < 1024]
    if to_download:
        with ThreadPool(num_workers) as pool:
            for _ in tqdm(pool.imap_unordered(download, to_download), total=len(to_download), desc='Downloading images'):
                pass

    # 构建样本列表，并显示进度
    samples = []
    for id_str, info in tqdm(img_info.items(), desc='Preparing samples'):
        img_id = int(id_str)
        file_path = os.path.join(images_dir, info['file_name'])
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids)
        anns = coco.loadAnns(ann_ids)

        # 合并实例掩码，处理多维或二维mask
        h, w = info['height'], info['width']
        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in anns:
            if 'segmentation' in ann:
                rle = maskUtils.frPyObjects(ann['segmentation'], h, w)
                m = maskUtils.decode(rle)
                if m.ndim == 2:
                    bin_mask = m.astype(np.uint8)
                else:
                    bin_mask = m.any(axis=2).astype(np.uint8)
                mask |= bin_mask

        # 提取关键点和可见性
        kps, vis = [], []
        for ann in anns:
            if 'keypoints' in ann:
                arr = np.array(ann['keypoints']).reshape(-1, 3)
                kps.extend(arr[:, :2].tolist())
                vis.extend(arr[:, 2].tolist())

        samples.append({
            'image_path': file_path,
            'mask': mask,
            'keypoints': np.array(kps, dtype=np.float32),
            'visibility': np.array(vis, dtype=np.int32),
        })

    return samples


class COCODataset(Dataset):
    """
    PyTorch Dataset for COCO 样本，带固定大小变换，兼容 UNet。
    """

    def __init__(self, samples, img_transform=None, mask_transform=None):
        self.samples = samples
        self.img_transform = img_transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        self.mask_transform = mask_transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = Image.open(item['image_path']).convert('RGB')
        img_tensor = self.img_transform(img)

        mask_np = item['mask']
        mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))
        mask_tensor = self.mask_transform(mask_img)
        mask_tensor = (mask_tensor > 0.5).float()

        kps = item.get('keypoints', None)
        vis = item.get('visibility', None)
        if kps is None or (isinstance(kps, np.ndarray) and kps.size == 0):
            kps_tensor = torch.zeros((0, 2), dtype=torch.float32)
            vis_tensor = torch.zeros((0,), dtype=torch.int32)
        else:
            coords = kps.astype(np.float32).copy()
            coords[:, 0] = coords[:, 0] / mask_np.shape[1] * 256
            coords[:, 1] = coords[:, 1] / mask_np.shape[0] * 256
            kps_tensor = torch.from_numpy(coords)
            vis_tensor = torch.from_numpy(vis.astype(np.int32))

        return img_tensor, mask_tensor, kps_tensor, vis_tensor, item['image_path']


def collate_fn(batch):
    imgs = torch.stack([x[0] for x in batch])
    masks = torch.stack([x[1] for x in batch])
    keypoints = [x[2] for x in batch]
    visibilities = [x[3] for x in batch]
    paths = [x[4] for x in batch]
    return imgs, masks, keypoints, visibilities, paths
