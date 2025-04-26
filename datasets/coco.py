import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO

from utils import transforms

class COCOSegmentationDataset(Dataset):
    """COCO Dataset for segmentation (person vs background). Returns full image and person mask."""
    def __init__(self, images_dir, ann_file, image_size, transform=True):
        super().__init__()
        self.coco = COCO(ann_file)
        # Get all image ids that have persons
        self.img_ids = self.coco.getImgIds(catIds=[1])  # COCO person category id is 1
        self.images_dir = images_dir
        self.image_size = image_size
        self.transform = transform

        # Prepare annotations: for each image, combine all person masks into one binary mask
        # (We could also prepare on the fly in __getitem__ for memory efficiency.)
        self.masks = {}
        for img_id in self.img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=[1], iscrowd=False)
            anns = self.coco.loadAnns(ann_ids)
            # Initialize empty mask
            img_info = self.coco.loadImgs([img_id])[0]
            mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
            for ann in anns:
                if 'segmentation' in ann:
                    # Merge each person mask (could be polygon or RLE) into one mask
                    rle = self.coco.annToRLE(ann)
                    m = self.coco.annToMask(ann)  # annToMask returns binary mask for this ann
                    mask = np.logical_or(mask, m).astype(np.uint8)
            self.masks[img_id] = mask

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        # Load image
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = self.masks[img_id].copy()

        # Resize image and mask to configured size
        image, mask = transforms.resize_image_and_mask(image, mask, self.image_size)
        if self.transform:
            # Random horizontal flip
            image, mask, _, _ = transforms.random_flip(image, mask)
            # (Optional: could add random crop, but we already resized to square)
            # Normalize image
        image = transforms.normalize_image(image)
        # Convert to tensor
        image_tensor = transforms.to_tensor(image)
        mask_tensor = transforms.to_tensor(mask.astype(np.uint8))
        return image_tensor, mask_tensor


class COCOKeypointsDataset(Dataset):
    """COCO Dataset for pose estimation (top-down per person). Each sample is a person crop with keypoint heatmap."""
    def __init__(self, images_dir, ann_file, is_train=True):
        super().__init__()
        self.coco = COCO(ann_file)
        # Get person category id (should be 1 for COCO)
        self.cat_id = 1
        # Prepare list of person annotations
        self.samples = []  # will hold tuples of (image_path, person_bbox, keypoints, visible)
        img_ids = self.coco.getImgIds(catIds=[self.cat_id])
        for img_id in img_ids:
            img_info = self.coco.loadImgs([img_id])[0]
            ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=[self.cat_id], iscrowd=False)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                # Each ann is one person
                if 'keypoints' not in ann:
                    continue
                keypoints = np.array(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
                # COCO keypoints format: [x1,y1,v1,... xk,yk,vk] where v indicates visibility (0=not in image, 1=visible, 2=occluded)
                coords = keypoints[:, :2]
                vis = (keypoints[:, 2] > 0).astype(np.uint8)  # treat both occluded and visible as 1, only 0 as not in image
                # Determine person bounding box
                if 'bbox' in ann:
                    # COCO bbox is [x, y, width, height] in original image coords
                    x, y, w, h = ann['bbox']
                    # Expand the bbox slightly for better context (e.g., 20% padding)
                    pad_w = 0.1 * w
                    pad_h = 0.1 * h
                    x1 = max(0, x - pad_w)
                    y1 = max(0, y - pad_h)
                    x2 = min(img_info['width'], x + w + pad_w)
                    y2 = min(img_info['height'], y + h + pad_h)
                else:
                    # If no bbox provided (should not happen in COCO person_keypoints), compute from keypoints
                    x1, y1 = coords.min(axis=0)
                    x2, y2 = coords.max(axis=0)
                    # add a 10% padding
                    pad_w = 0.1 * (x2 - x1)
                    pad_h = 0.1 * (y2 - y1)
                    x1 = max(0, x1 - pad_w); y1 = max(0, y1 - pad_h)
                    x2 = min(img_info['width'], x2 + pad_w); y2 = min(img_info['height'], y2 + pad_h)
                bbox = [x1, y1, x2, y2]
                img_path = os.path.join(images_dir, img_info['file_name'])
                self.samples.append((img_path, img_info['width'], img_info['height'], bbox, coords, vis))
        self.is_train = is_train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, img_w, img_h, bbox, keypoints, vis = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x1, y1, x2, y2 = map(int, bbox)
        # Crop the image to the person bounding box
        crop = image[y1:y2, x1:x2, :].copy()
        crop_h, crop_w = crop.shape[:2]
        # Resize crop to target POSE_CROP_SIZE
        target_w, target_h = transforms.POSE_CROP_SIZE
        crop_resized = cv2.resize(crop, (target_w, target_h))
        # Adjust keypoints to crop coordinate and new scale
        scale_x = target_w / crop_w
        scale_y = target_h / crop_h
        keypoints_adj = keypoints.copy()
        keypoints_adj[:, 0] = (keypoints_adj[:, 0] - x1) * scale_x
        keypoints_adj[:, 1] = (keypoints_adj[:, 1] - y1) * scale_y

        # Data augmentation
        if self.is_train:
            crop_resized, _, keypoints_adj, vis = transforms.random_flip(crop_resized, None, keypoints_adj, vis)
            crop_resized, _, keypoints_adj, vis = transforms.random_rotate_and_scale(crop_resized, None, keypoints_adj, vis)
        # Normalize image
        crop_norm = transforms.normalize_image(crop_resized)
        # Generate heatmaps
        heatmaps = transforms.generate_heatmaps(keypoints_adj, vis, output_size=(target_h//4, target_w//4))
        # Convert to tensor
        image_tensor = transforms.to_tensor(crop_norm).float()
        heatmaps_tensor = transforms.to_tensor(heatmaps).float()
        return image_tensor, heatmaps_tensor
