import os
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
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
from pycocotools.coco import COCO


def prepare_coco_dataset(root_dir,
                         split='train',
                         max_samples=None,
                         force_reload=False,
                         num_workers=8):
    """
    下载并准备 COCO 2017 数据集，返回包含 person segmentation + keypoints 的样本列表。
    Args:
        root_dir (str): 数据下载与缓存根目录
        split (str): 'train' 或 'val'
        max_samples (int, optional): 最多返回多少样本；默认不限制
        force_reload (bool): 若为 True，强制重建缓存；否则有缓存直接加载
        num_workers (int): 多线程数，用于加速样本处理
    Returns:
        List[Dict]: 每个 dict 包含：
            {
                'image_path': str,
                'mask': np.ndarray (H×W),
                'keypoints': np.ndarray (M×2),
                'visibility': np.ndarray (M,)
            }
    """
    # —— URL & 目录配置 —— #
    IMG_URL = f'http://images.cocodataset.org/zips/{split}2017.zip'
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
        os.makedirs(os.path.dirname(target_zip), exist_ok=True)
        if check_file and os.path.isfile(check_file) and not force_reload:
            return
        for attempt in range(1, max_retries+1):
            if not os.path.isfile(target_zip) or not zipfile.is_zipfile(target_zip):
                urllib.request.urlretrieve(url, target_zip)
            try:
                with zipfile.ZipFile(target_zip, 'r') as z:
                    if check_file and not os.path.isfile(check_file):
                        z.extractall(os.path.dirname(extract_to))
                    break
            except zipfile.BadZipFile:
                os.remove(target_zip)
                if attempt == max_retries:
                    raise RuntimeError(f"Failed to download {url}")

    # 下载并解压 images & annotations
    sample_img = os.path.join(img_dir, '000000000009.jpg')
    download_and_extract(IMG_URL,
                         os.path.join(root_dir, f'{split}2017.zip'),
                         img_dir, check_file=sample_img)
    kp_json = os.path.join(anno_dir, f'person_keypoints_{split}2017.json')
    download_and_extract(ANNO_URL,
                         os.path.join(root_dir, 'annotations_trainval2017.zip'),
                         anno_dir, check_file=kp_json)

    inst_file = os.path.join(anno_dir, f'instances_{split}2017.json')
    kp_file   = os.path.join(anno_dir, f'person_keypoints_{split}2017.json')
    for f in (inst_file, kp_file):
        if not os.path.isfile(f):
            raise FileNotFoundError(f"Missing file: {f}")

    # —— 缓存加载 —— #
    if os.path.isfile(cache_file) and not force_reload:
        with open(cache_file, 'rb') as fp:
            samples = pickle.load(fp)
        if max_samples:
            samples = samples[:max_samples]
        return samples

    # —— 解析 COCO 注释 —— #
    coco_seg = COCO(inst_file)
    coco_kp  = COCO(kp_file)
    seg_ids = coco_seg.getAnnIds(catIds=[1], iscrowd=None)
    kp_ids  = set(coco_kp.getAnnIds(catIds=[1]))

    # 按 image_id 聚合 ann
    ann_map = {}
    for ann in tqdm(coco_seg.loadAnns(seg_ids), desc=f"Building ann_map ({split})"):
        aid = ann['id']
        if aid in kp_ids:
            iid = ann['image_id']
            ann_map.setdefault(iid, []).append(ann)

    image_ids = list(ann_map.keys())
    if max_samples:
        image_ids = image_ids[:max_samples]

    # 样本处理函数
    def process_image(iid):
        anns = ann_map[iid]
        img_info = coco_seg.loadImgs(iid)[0]
        H, W = img_info['height'], img_info['width']
        mask = np.zeros((H, W), dtype=np.uint8)
        kps_list, vis_list = [], []
        for ann in anns:
            m = coco_seg.annToMask(ann)
            mask = np.maximum(mask, m)
            ann_kp = coco_kp.loadAnns([ann['id']])[0]
            kp = np.array(ann_kp['keypoints'], dtype=np.float32).reshape(-1,3)
            kps_list.append(kp[:,:2])
            vis_list.append(kp[:,2].astype(np.int32))
        if kps_list:
            keypoints  = np.vstack(kps_list)
            visibility = np.concatenate(vis_list)
        else:
            keypoints  = np.zeros((0,2), dtype=np.float32)
            visibility = np.zeros((0,),  dtype=np.int32)
        return {
            'image_path': os.path.join(img_dir, img_info['file_name']),
            'mask': mask,
            'keypoints': keypoints,
            'visibility': visibility
        }

    # 多线程并行处理并显示进度
    with ThreadPool(num_workers) as pool:
        samples = list(tqdm(pool.imap(process_image, image_ids),
                            total=len(image_ids),
                            desc=f"Processing {split} images"))

    # 缓存到磁盘
    mkdir_p(os.path.dirname(cache_file))
    with open(cache_file, 'wb') as fp:
        pickle.dump(samples, fp, protocol=pickle.HIGHEST_PROTOCOL)


    print(f"[prepare_coco_dataset] split={split}, loaded {len(samples)} samples")
    return samples

# COCO 骨骼对 (0-based)
SKELETON = [
    (15,13),(13,11),(16,14),(14,12),(11,12),
    (5,11),(6,12),(5,6),(5,7),(7,9),
    (6,8),(8,10),(1,2),(0,1),(0,2),
    (1,3),(2,4),(3,5),(4,6)
]


def _gaussian_kernel(size, sigma=2):
    """生成 size×size 的高斯核"""
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / kernel.max()


def generate_heatmap(coords, vis, K, H, W, sigma=2):
    """
    coords: [P,17,2] 原始坐标
    vis:    [P,17] 可见性
    输出 heatmap: [17, H, W]
    """
    hm = np.zeros((K, H, W), dtype=np.float32)
    tmp_size = sigma * 3
    for p in range(coords.shape[0]):
        for k in range(K):
            if vis[p, k] <= 0:
                continue
            x, y = coords[p, k]
            x = int(x / W * W)
            y = int(y / H * H)
            ul = [int(x - tmp_size), int(y - tmp_size)]
            br = [int(x + tmp_size + 1), int(y + tmp_size + 1)]
            if ul[0] >= W or ul[1] >= H or br[0] < 0 or br[1] < 0:
                continue
            gk = _gaussian_kernel(tmp_size*2+1, sigma)
            g_h, g_w = gk.shape
            g_x = max(0, -ul[0]), min(br[0], W) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], H) - ul[1]
            img_x = max(0, ul[0]), min(br[0], W)
            img_y = max(0, ul[1]), min(br[1], H)
            hm[k, img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
                hm[k, img_y[0]:img_y[1], img_x[0]:img_x[1]],
                gk[g_y[0]:g_y[1], g_x[0]:g_x[1]]
            )
    return hm


def generate_paf(coords, vis, skeleton, H, W, paf_thickness=1):
    """
    coords: [P,17,2]
    vis:    [P,17]
    skeleton: list of (a,b)
    输出 paf: [2L, H, W]
    """
    L = len(skeleton)
    paf = np.zeros((2*L, H, W), dtype=np.float32)
    count = np.zeros((L, H, W), dtype=np.int32)
    for p in range(coords.shape[0]):
        for idx, (a, b) in enumerate(skeleton):
            if vis[p, a] <= 0 or vis[p, b] <= 0:
                continue
            x1, y1 = coords[p, a]
            x2, y2 = coords[p, b]
            # 线段向量
            vx = x2 - x1; vy = y2 - y1
            norm = np.sqrt(vx*vx + vy*vy)
            if norm < 1e-4:
                continue
            vx /= norm; vy /= norm
            # 遍历所有像素，判断距线段小于阈值
            # 简化: 对整个 bbox 进行填充
            xmin = int(max(min(x1, x2) - paf_thickness, 0))
            xmax = int(min(max(x1, x2) + paf_thickness, W))
            ymin = int(max(min(y1, y2) - paf_thickness, 0))
            ymax = int(min(max(y1, y2) + paf_thickness, H))
            for y in range(ymin, ymax):
                for x in range(xmin, xmax):
                    # 投影距离
                    dx = x - x1; dy = y - y1
                    proj = dx*vx + dy*vy
                    if proj < 0 or proj > norm:
                        continue
                    # 到线段的垂直距离
                    perp = abs(-vy*dx + vx*dy)
                    if perp <= paf_thickness:
                        paf[2*idx    , y, x] += vx
                        paf[2*idx + 1, y, x] += vy
                        count[idx, y, x] += 1
    # 平均化
    for idx in range(L):
        mask = count[idx] > 0
        paf[2*idx    ][mask] /= count[idx][mask]
        paf[2*idx + 1][mask] /= count[idx][mask]
    return paf


class COCODataset(Dataset):
    """
    COCO Dataset for Bottom-Up multi-person pose + segmentation.
    返回:
      img_tensor:      [3,H,W]
      mask_tensor:     [1,H,W]
      hm_label:        [17, H', W']
      paf_label:       [2L, H', W']
      keypoints:       [P,17,2] 原始坐标
      visibility:      [P,17]
      path:            str
    """
    def __init__(self, samples, img_transform=None, mask_transform=None,
                 hm_size=(64,64), paf_size=(64,64), sigma=2, paf_thickness=1):
        self.samples = samples
        self.img_transform = img_transform or transforms.Compose([
            transforms.Resize((480,480)), transforms.ToTensor()
        ])
        self.mask_transform = mask_transform or transforms.Compose([
            transforms.Resize((480,480)), transforms.ToTensor()
        ])
        self.hm_size = hm_size
        self.paf_size = paf_size
        self.sigma = sigma
        self.paf_thickness = paf_thickness

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = Image.open(item['image_path']).convert('RGB')
        img_tensor = self.img_transform(img)
        # seg mask
        mask = Image.fromarray((item['mask']*255).astype(np.uint8))
        mask_tensor = self.mask_transform(mask)
        mask_tensor = (mask_tensor > 0.5).float()
        # 原始 keypoints + vis
        coords = item['keypoints']    # [M,2]
        vis    = item['visibility']   # [M]
        # reshape to [P,17,2] and [P,17]
        P = coords.shape[0] // 17
        kps = coords.reshape(P,17,2)
        vis = vis.reshape(P,17)
        # 生成 heatmap & paf label
        hm_label  = generate_heatmap(kps, vis, 17,
                                     self.hm_size[0], self.hm_size[1],
                                     sigma=self.sigma)
        paf_label = generate_paf(kps, vis, SKELETON,
                                 self.paf_size[0], self.paf_size[1],
                                 paf_thickness=self.paf_thickness)
        # to tensor
        hm_label  = torch.from_numpy(hm_label)
        paf_label = torch.from_numpy(paf_label)
        return img_tensor, mask_tensor, hm_label, paf_label, kps, vis, item['image_path']


def collate_fn(batch):
    imgs       = torch.stack([x[0] for x in batch], dim=0)
    masks      = torch.stack([x[1] for x in batch], dim=0)
    hm_labels  = torch.stack([x[2] for x in batch], dim=0)
    paf_labels = torch.stack([x[3] for x in batch], dim=0)
    kps_list   = [x[4] for x in batch]
    vis_list   = [x[5] for x in batch]
    paths      = [x[6] for x in batch]
    return imgs, masks, hm_labels, paf_labels, kps_list, vis_list, paths
