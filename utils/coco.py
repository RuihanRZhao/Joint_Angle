import os
import numpy as np
import torch
from pycocotools.coco import COCO
from PIL import Image
import zipfile
from typing import Tuple, List
from tqdm import tqdm
import requests
from torch.utils.data import Dataset
import time
import math

COCO_PERSON_SKELETON: List[Tuple[int, int]] = [
    (15, 13), (13, 11), (16, 14), (14, 12),
    (5, 11), (6, 12), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1),
    (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)
]

FLIP_INDEX = [
    0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15
]

NUM_KP = 17
NUM_LIMBS = len(COCO_PERSON_SKELETON)


class COCOPoseDataset(Dataset):
    """
    COCO 2017 人体关键点姿态数据集
    - 自动下载（如果缺失）
    - 过滤无标注图像
    - 生成热图（heatmap）和 PAF（part affinity field）
    """

    def __init__(
        self,
        root: str,
        img_dir: str,
        ann_file: str,
        input_size: tuple = (256, 256),
        hm_size: tuple = (64, 64),
        sigma: float = 2.0,
        paf_thickness: int = 1,
        download_retries: int = 3,
        backoff_factor: float = 2.0,
    ):
        """
        Args:
            root (str): 数据根目录，例如 "data/coco"
            img_dir (str): 子目录名，如 "train2017" 或 "val2017"
            ann_file (str): 注解文件名，如 "person_keypoints_train2017.json"
            input_size (tuple): 模型输入尺寸 (H, W)
            hm_size (tuple): 热图输出尺寸 (H, W)
            sigma (float): 关键点热图高斯半径
            paf_thickness (int): PAF 线宽度
            download_retries (int): 数据下载最大重试次数
            backoff_factor (float): 重试等待倍增因子
        """
        super().__init__()

        # 1) 确保 COCO 数据存在
        ensure_coco_data(root, retries=download_retries, backoff_factor=backoff_factor)

        # 2) 加载 COCO 注解
        ann_path = os.path.join(root, ann_file)
        if not os.path.isfile(ann_path):
            raise RuntimeError(f"找不到注解文件：{ann_path}")
        self.coco = COCO(ann_path)

        # 3) 图像目录检查
        self.img_dir = os.path.join(root, img_dir)
        if not os.path.isdir(self.img_dir):
            raise RuntimeError(f"找不到图像目录：{self.img_dir}")

        # 4) 筛选包含关键点的图像 ID
        all_ids = list(self.coco.imgs.keys())
        self.img_ids = [
            img_id for img_id in all_ids
            if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=False,catIds=[1])) > 0
        ]
        # 空集检查
        if len(self.img_ids) == 0:
            raise RuntimeError(f"数据集中没有找到任何包含关键点的图像，请检查 {ann_path} 的内容。")

        # 5) 参数存储
        self.img_size = input_size
        self.hm_size = hm_size
        self.sigma = sigma
        self.paf_thickness = paf_thickness

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        """
        Returns:
            img_tensor: FloatTensor, [3, H, W], 归一化到 [0,1]
            hm_tensor:  FloatTensor, [K, H_hm, W_hm]
            paf_tensor: FloatTensor, [2*C, H_hm, W_hm]
            n_person:   int, 该图像中人的数量
        """
        img_id = self.img_ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, info['file_name'])
        img = Image.open(img_path).convert("RGB")
        img = img.resize(self.img_size, Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

        # 加载标注
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False, category_id=1)
        anns = self.coco.loadAnns(ann_ids)
        n_person = len(anns)

        # 生成热图和 PAF
        hm_tensor = generate_heatmap(anns, self.hm_size, self.sigma)
        paf_tensor = generate_paf(anns, self.hm_size, COCO_PERSON_SKELETON, self.paf_thickness)

        return img_tensor, hm_tensor, paf_tensor, n_person



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
                z.extractall(root)
        except zipfile.BadZipFile as e:
            raise RuntimeError(f"解压 {zip_path} 失败：{e}")

        # 删除 zip 文件，节省空间
        try:
            os.remove(zip_path)
        except OSError:
            pass

def generate_heatmap(anns, input_size, hm_size, sigma):
    """
    生成关键点热图 (heatmap)。

    Args:
        anns (list): COCO 注释列表，每个 ann 包含 'keypoints'，格式 [x1,y1,v1,...,x17,y17,v17]
        input_size (tuple): 输入图像大小 (H, W)
        hm_size (tuple): 热图输出大小 (H_hm, W_hm)
        sigma (float): 高斯标准差

    Returns:
        torch.FloatTensor: 形状 [K, H_hm, W_hm] 的热图张量
    """
    H, W = input_size
    hm_H, hm_W = hm_size
    num_keypoints = 17
    heatmaps = np.zeros((num_keypoints, hm_H, hm_W), dtype=np.float32)
    tmp_size = sigma * 3

    for ann in anns:
        kp = ann['keypoints']
        for idx in range(num_keypoints):
            x, y, v = kp[3*idx], kp[3*idx+1], kp[3*idx+2]
            if v <= 0:
                continue

            # 映射到热图分辨率
            x_hm = x * hm_W / W
            y_hm = y * hm_H / H

            # 高斯核覆盖区域
            ul = [int(x_hm - tmp_size), int(y_hm - tmp_size)]
            br = [int(x_hm + tmp_size) + 1, int(y_hm + tmp_size) + 1]
            if ul[0] >= hm_W or ul[1] >= hm_H or br[0] < 0 or br[1] < 0:
                continue

            size = int(2 * tmp_size + 1)
            x_range = np.arange(0, size, 1, np.float32)
            y_range = x_range[:, np.newaxis]
            gaussian = np.exp(-((x_range - tmp_size)**2 + (y_range - tmp_size)**2) / (2 * sigma * sigma))

            g_x0 = max(0, -ul[0])
            g_y0 = max(0, -ul[1])
            img_x0 = max(0, ul[0])
            img_y0 = max(0, ul[1])
            g_x1 = min(br[0], hm_W) - ul[0]
            g_y1 = min(br[1], hm_H) - ul[1]
            img_x1 = min(br[0], hm_W)
            img_y1 = min(br[1], hm_H)

            heatmaps[idx,
                     img_y0:img_y1,
                     img_x0:img_x1] = np.maximum(
                         heatmaps[idx, img_y0:img_y1, img_x0:img_x1],
                         gaussian[g_y0:g_y1, g_x0:g_x1]
                     )

    return torch.from_numpy(heatmaps)


def generate_paf(anns, input_size, hm_size, skeleton, paf_thickness):
    """
    生成部件亲和场 (PAF)。

    Args:
        anns (list): COCO 注释列表，每个 ann 包含 'keypoints'
        input_size (tuple): 输入图像大小 (H, W)
        hm_size (tuple): PAF 输出大小 (H_hm, W_hm)
        skeleton (list of [int, int]): 骨架连接点索引对（COCO 索引从 1 开始）
        paf_thickness (float): PAF 半宽度（像素）

    Returns:
        torch.FloatTensor: 形状 [2*C, H_hm, W_hm] 的 PAF 张量
    """
    H, W = input_size
    hm_H, hm_W = hm_size
    num_limbs = len(skeleton)
    pafs = np.zeros((2 * num_limbs, hm_H, hm_W), dtype=np.float32)
    count = np.zeros((num_limbs, hm_H, hm_W), dtype=np.int32)

    for ann in anns:
        kp = ann['keypoints']
        for idx, (p1, p2) in enumerate(skeleton):
            x1, y1, v1 = kp[(p1-1)*3], kp[(p1-1)*3+1], kp[(p1-1)*3+2]
            x2, y2, v2 = kp[(p2-1)*3], kp[(p2-1)*3+1], kp[(p2-1)*3+2]
            if v1 <= 0 or v2 <= 0:
                continue

            # 映射到 PAF 分辨率
            x1_hm = x1 * hm_W / W
            y1_hm = y1 * hm_H / H
            x2_hm = x2 * hm_W / W
            y2_hm = y2 * hm_H / H

            dx, dy = x2_hm - x1_hm, y2_hm - y1_hm
            norm = math.hypot(dx, dy)
            if norm < 1e-4:
                continue
            vx, vy = dx / norm, dy / norm

            # 搜索区域
            min_x = int(max(min(x1_hm, x2_hm) - paf_thickness, 0))
            max_x = int(min(max(x1_hm, x2_hm) + paf_thickness, hm_W))
            min_y = int(max(min(y1_hm, y2_hm) - paf_thickness, 0))
            max_y = int(min(max(y1_hm, y2_hm) + paf_thickness, hm_H))

            for y in range(min_y, max_y):
                for x in range(min_x, max_x):
                    rx, ry = x - x1_hm, y - y1_hm
                    proj = rx * vx + ry * vy
                    if proj <= 0 or proj >= norm:
                        continue
                    perp_dist = abs(-vy * rx + vx * ry)
                    if perp_dist > paf_thickness:
                        continue
                    pafs[2*idx, y, x] += vx
                    pafs[2*idx+1, y, x] += vy
                    count[idx, y, x] += 1

    # 平均化向量
    for idx in range(num_limbs):
        mask = count[idx] > 0
        pafs[2*idx][mask] /= count[idx][mask]
        pafs[2*idx+1][mask] /= count[idx][mask]

    return torch.from_numpy(pafs)