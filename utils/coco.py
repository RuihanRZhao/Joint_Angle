import numpy as np
import fiftyone as fo
import fiftyone.zoo as foz

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


def prepare_coco_dataset(root_dir, split, max_samples=None):
    """
    加载 COCO-2017 数据集并提取样本列表，保证分割 mask 重构为原图尺寸。

    root_dir: 数据根目录，用于 fo.config.dataset_zoo_dir
    split: 'train' 或 'validation'
    max_samples: 最大样本量，None 表示全量

    返回:
        List of dicts, each 包含:
            image_path (str)
            mask (np.ndarray) 二值seg掩码 (H, W)
            keypoints (np.ndarray) 关键点坐标 (M,2)
            visibility (np.ndarray) 可见性 (M,)
    """
    # 指定下载目录
    fo.config.dataset_zoo_dir = root_dir

    print(f"[DEBUG] Loading COCO-2017 split={split}, max_samples={max_samples}")
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split=split,
        label_types=["segmentations", "keypoints"],
        classes=["person"],
        max_samples=max_samples,
    )
    print(f"[DEBUG] FiftyOne loaded {len(dataset)} samples")

    samples = []
    for idx, sample in enumerate(dataset):
        img_path = sample.filepath
        # 图像原始尺寸
        meta = sample.metadata
        H, W = meta.height, meta.width

        # 获取标注
        seg_field = getattr(sample, 'segmentations', None)
        kp_field = getattr(sample, 'keypoints', None)
        seg_dets = seg_field.detections if seg_field else []
        kp_groups = kp_field.keypoints if kp_field else []

        if not seg_dets:
            print(f"[DEBUG] sample {idx} 无 segmentation，跳过")
            continue

        combined_mask = None
        # 合并所有实例掩码，重构为(full H,W)
        for det in seg_dets:
            mask = det.mask
            if mask is None:
                continue
            m = np.array(mask, dtype=bool)
            # 计算det在原图中的位置
            bbox = det.bounding_box  # [x_rel, y_rel, w_rel, h_rel]
            x0 = int(bbox[0] * W)
            y0 = int(bbox[1] * H)
            h_m, w_m = m.shape
            # 防止超出边界
            if y0 + h_m > H or x0 + w_m > W:
                print(f"[WARNING] mask shape/box超出图像边界, idx={idx}")
                h_m = min(h_m, H - y0)
                w_m = min(w_m, W - x0)
                m = m[:h_m, :w_m]
            full_mask = np.zeros((H, W), dtype=bool)
            full_mask[y0:y0+h_m, x0:x0+w_m] = m
            combined_mask = full_mask if combined_mask is None else (combined_mask | full_mask)

        if combined_mask is None:
            print(f"[DEBUG] sample {idx} segmentation mask 全空，跳过")
            continue

        # 提取所有 keypoints
        all_kps = []
        all_vis = []
        for group in kp_groups:
            arr = np.array(group, dtype=np.float32).reshape(-1, 3)
            coords = arr[:, :2]
            vis = arr[:, 2]
            for (x, y), v in zip(coords, vis):
                all_kps.append([float(x), float(y)])
                all_vis.append(int(v))
        keypoints_arr = np.array(all_kps, dtype=np.float32).reshape(-1, 2) if all_kps else np.zeros((0,2), dtype=np.float32)
        visibility_arr = np.array(all_vis, dtype=np.int32) if all_vis else np.zeros((0,), dtype=np.int32)

        samples.append({
            'image_path': img_path,
            'mask': combined_mask.astype(np.uint8),
            'keypoints': keypoints_arr,
            'visibility': visibility_arr,
        })

    print(f"[DEBUG] Prepared {len(samples)} valid samples for split={split}")
    return samples


class COCODataset(Dataset):
    """
    PyTorch Dataset for COCO samples, with resizing to fixed dimensions for UNet compatibility.
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
