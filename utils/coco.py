import os
import json
import requests
from pycocotools import mask as maskUtils
from multiprocessing.pool import ThreadPool
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

import errno
import urllib.request
import zipfile
import pickle
from multiprocessing.dummy import Pool as ThreadPool

import numpy as np
from PIL import Image
from pycocotools.coco import COCO

def prepare_coco_dataset(root_dir,
                         split='train',
                         max_samples=None,
                         force_reload=False,
                         num_workers=8):
    """
    下载并准备 COCO 2017 数据集，返回包含 person segmentation + keypoints 的样本列表。

    Args:
        root_dir (str): 数据下载与缓存根目录，最终结构为
            root_dir/
                images/train2017/
                images/val2017/
                annotations/
        split (str): 'train' 或 'val'
        max_samples (int, optional): 最多返回多少样本；默认不限制
        force_reload (bool): 若为 True，强制重建缓存；否则有缓存直接加载
        num_workers (int): 多线程数，用于加速样本处理

    Returns:
        List[Dict]: 每个 dict 包含：
            {
                'image_path': str,
                'mask': np.ndarray (H×W 二值 mask),
                'keypoints': np.ndarray (M×2 float32),
                'visibility': np.ndarray (M int32)
            }
    """
    # ———— 1. URL & 目录配置 ————
    IMG_URL = 'http://images.cocodataset.org/zips/{}2017.zip'.format(split)
    ANNO_URL = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    img_dir = os.path.join(root_dir, 'images', f'{split}2017')
    anno_dir = os.path.join(root_dir, 'annotations')
    cache_file = os.path.join(root_dir, f'coco_{split}_samples.pkl')

    def mkdir_p(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    def download_and_extract(url, target_zip, extract_to, check_file=None, max_retries=3):
        """
        下载并解压 ZIP，带进度条，并对损坏的 ZIP 自动重试下载。

        Args:
            url (str): 远程 zip 地址
            target_zip (str): 本地存储 zip 的路径
            extract_to (str): 解压目标根目录
            check_file (str, optional): 解压后应该存在的某个文件（绝对路径），用来判断是否已经解压过
            max_retries (int): ZIP 损坏时最大重试次数
        """

        def _reporthook(block_num, block_size, total_size):
            if pbar.total is None:
                pbar.total = total_size
            downloaded = block_num * block_size
            pbar.update(block_size)

        # 确保目录存在
        os.makedirs(os.path.dirname(target_zip), exist_ok=True)

        # 如果目标解压文件已存在，跳过下载和解压
        if check_file and os.path.isfile(check_file):
            return

        # 下载并校验
        for attempt in range(1, max_retries + 1):
            # 如果本地没有 ZIP，或上一次下载后被判定为坏包，就（重新）下载
            need_download = True
            if os.path.isfile(target_zip):
                # 快速检测：文件必须是合法的 ZIP
                if zipfile.is_zipfile(target_zip):
                    need_download = False
                else:
                    print(f"[Warning] 已下载的 {os.path.basename(target_zip)} 不是合法 ZIP，删除并重试")
                    os.remove(target_zip)

            if need_download:
                with tqdm(
                        desc=f"Downloading {os.path.basename(target_zip)} (attempt {attempt})",
                        unit="B", unit_scale=True, unit_divisor=1024
                ) as pbar:
                    urllib.request.urlretrieve(url, target_zip, reporthook=_reporthook)

            # 尝试打开测试
            try:
                with zipfile.ZipFile(target_zip, 'r') as z:
                    # 简单列出首个文件以触发惰性解析
                    z.namelist()[:1]
            except zipfile.BadZipFile:
                print(f"[Error] {os.path.basename(target_zip)} 解析失败（BadZipFile），将重试下载")
                os.remove(target_zip)
                if attempt == max_retries:
                    raise RuntimeError(f"下载 {url} 后多次校验失败，请检查网络或手动下载。")
                continue
            else:
                # 如果只做校验，可以在这里返回；否则继续到“解压”步骤
                break

        # 解压（如果指定了 check_file，就只在它不存在时解压）
        need_extract = True
        if check_file:
            need_extract = not os.path.isfile(check_file)

        if need_extract:
            print(f"Extracting {os.path.basename(target_zip)} …")
            with zipfile.ZipFile(target_zip, 'r') as z:
                z.extractall(os.path.dirname(extract_to))

    # ———— 2. 下载 & 校验 ————
    # 确保 images/train2017 下至少有一张样本图
    sample_img = os.path.join(img_dir, '000000000009.jpg')
    download_and_extract(
        IMG_URL,
        os.path.join(root_dir, f'{split}2017.zip'),
        img_dir,
        check_file=sample_img
    )

    # 确保 annotations 下有 person_keypoints_train2017.json
    kp_json = os.path.join(anno_dir, f'person_keypoints_{split}2017.json')
    download_and_extract(
        ANNO_URL,
        os.path.join(root_dir, 'annotations_trainval2017.zip'),
        anno_dir,
        check_file=kp_json
    )

    inst_file = os.path.join(anno_dir, f'instances_{split}2017.json')
    kp_file   = os.path.join(anno_dir, f'person_keypoints_{split}2017.json')
    for f in (inst_file, kp_file):
        if not os.path.isfile(f):
            raise FileNotFoundError(f"找不到文件：{f}")

    # ———— 3. 缓存加载 ————
    if os.path.isfile(cache_file) and not force_reload:
        with open(cache_file, 'rb') as fp:
            samples = pickle.load(fp)
        return samples

    # ———— 4. 解析 COCO 注释 ————
    coco_seg = COCO(inst_file)
    coco_kp  = COCO(kp_file)
    # 只保留 category_id=1 (person) 的注释 ID
    seg_ids = coco_seg.getAnnIds(catIds=[1], iscrowd=None)
    kp_ids  = set(coco_kp.getAnnIds(catIds=[1]))

    # ① 按 image_id 聚合：只保留同时在 seg & kp 中出现的 ann_id
    ann_map = {}
    for ann in coco_seg.loadAnns(seg_ids):
        aid = ann['id']
        if aid in kp_ids:
            iid = ann['image_id']
            ann_map.setdefault(iid, []).append(ann)

    image_ids = list(ann_map.keys())
    if max_samples:
        image_ids = image_ids[:max_samples]

    # ———— 5. 样本处理函数 ————
    def process_image(iid):
        anns = ann_map[iid]
        # 构建全人物 mask
        img_info = coco_seg.loadImgs(iid)[0]
        H, W = img_info['height'], img_info['width']
        mask = np.zeros((H, W), dtype=np.uint8)
        kps_list = []
        vis_list = []

        for ann in anns:
            # segmentation mask
            m = coco_seg.annToMask(ann)  # 0/1 mask
            mask = np.maximum(mask, m)

            # keypoints
            ann_kp = coco_kp.loadAnns([ann['id']])[0]
            kp = np.array(ann_kp['keypoints'], dtype=np.float32).reshape(-1, 3)
            coords = kp[:, :2]       # (17,2)
            vis    = kp[:, 2].astype(np.int32)
            kps_list.append(coords)
            vis_list.append(vis)

        # 拼成 (M,2) & (M,)
        keypoints = np.vstack(kps_list)    if kps_list else np.zeros((0,2), dtype=np.float32)
        visibility = np.concatenate(vis_list) if vis_list else np.zeros((0,), dtype=np.int32)

        img_path = os.path.join(img_dir, img_info['file_name'])
        return {
            'image_path': img_path,
            'mask': mask,
            'keypoints': keypoints,
            'visibility': visibility
        }

    # ———— 6. 多线程加速 & 聚集 ————
    with ThreadPool(num_workers) as pool:
        samples = pool.map(process_image, image_ids)

    # ———— 7. 缓存到磁盘 ————
    mkdir_p(os.path.dirname(cache_file))
    with open(cache_file, 'wb') as fp:
        pickle.dump(samples, fp, protocol=pickle.HIGHEST_PROTOCOL)

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
            # no keypoints
            kps_resized = torch.zeros((0, 2), dtype=torch.float32)
            kps_orig    = torch.zeros((0, 2), dtype=torch.float32)
            vis_tensor  = torch.zeros((0,), dtype=torch.int32)
        else:
            # absolute original coords
            orig_coords = kps.astype(np.float32).copy()
            # resized coords for model input (256x256)
            resized = orig_coords.copy()
            resized[:, 0] = orig_coords[:, 0] / mask_np.shape[1] * 256
            resized[:, 1] = orig_coords[:, 1] / mask_np.shape[0] * 256
            kps_resized = torch.from_numpy(resized)
            kps_orig    = torch.from_numpy(orig_coords)
            vis_tensor  = torch.from_numpy(vis.astype(np.int32))

        return img_tensor, mask_tensor, kps_resized, vis_tensor, item['image_path'], kps_orig


def collate_fn(batch):
    imgs = torch.stack([x[0] for x in batch])
    masks = torch.stack([x[1] for x in batch])
    keypoints = [x[2] for x in batch]
    visibilities = [x[3] for x in batch]
    paths = [x[4] for x in batch]
    keypoints_orig = [x[5] for x in batch]
    return imgs, masks, keypoints, visibilities, paths, keypoints_orig


if __name__ == '__main__':
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description="COCO 数据加载 Demo")
    parser.add_argument('--data_dir',    type=str, default='../run/data', help='COCO 数据集根目录')
    parser.add_argument('--split',       type=str, default='train', help="'train' 或 'validation'")
    parser.add_argument('--max_samples', type=int, default=100, help='最多打印多少张样本')
    parser.add_argument('--batch_size',  type=int, default=2, help='DataLoader batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader num_workers')
    args = parser.parse_args()

    # 1. 准备样本列表
    print(f"Preparing up to {args.max_samples} samples from split '{args.split}' in {args.data_dir}...")
    samples = prepare_coco_dataset(
        root_dir=args.data_dir,
        split=args.split,
        max_samples=args.max_samples,
        force_reload=False,
        num_workers=args.num_workers
    )
    print(f"  -> Got {len(samples)} samples.")

    # 2. 构建 Dataset 和 DataLoader
    dataset = COCODataset(samples)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 3. 遍历一个 batch，打印信息
    for batch_idx, (imgs, masks, kps_list, vis_list, paths, kps_orig_list) in enumerate(loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  imgs.shape = {imgs.shape}")   # [B,3,256,256]
        print(f"  masks.shape = {masks.shape}") # [B,1,256,256]")
        for i, (kps, vis, path) in enumerate(zip(kps_list, vis_list, paths)):
            print(f"    sample {i}:")
            print(f"      path           = {path}")
            print(f"      num_keypoints  = {kps.shape[0]}")
            print(f"      visibility     = {vis.tolist()}")
        # 只打印第一个 batch 即可
        break
