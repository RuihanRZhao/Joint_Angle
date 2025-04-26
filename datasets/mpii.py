import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from utils import transforms

class MPIIPoseDataset(Dataset):
    """MPII Dataset for pose estimation. Each sample is an image (possibly cropped) with keypoints heatmap."""
    def __init__(self, images_dir, ann_list, is_train=True):
        """
        images_dir: path to MPII images
        ann_list: list of annotations, each a dict with keys 'image', 'keypoints', 'visible'
        is_train: whether to apply augmentation
        """
        super().__init__()
        self.images_dir = images_dir
        self.annotations = ann_list  # list of dicts
        self.is_train = is_train

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_path = os.path.join(self.images_dir, ann['image'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        keypoints = np.array(ann['keypoints'], dtype=np.float32)  # shape (16, 2)
        visible = np.array(ann.get('visible', [1]*len(keypoints)), dtype=np.uint8)  # some annotations include visibility

        # For MPII, assume the image already tightly frames the person (or a given scale is provided).
        # If needed, one could crop around the person using a provided scale or compute min/max of keypoints.
        # Here, we'll assume the whole image is the person region for simplicity.
        img_h, img_w = image.shape[:2]
        # Resize image to the pose input size (keeping aspect ratio by padding if needed)
        target_w, target_h = transforms.POSE_CROP_SIZE
        # We will scale so that the longer side fits, and pad the other side
        scale = min(target_w/img_w, target_h/img_h)
        new_w, new_h = int(img_w*scale), int(img_h*scale)
        image_resized = cv2.resize(image, (new_w, new_h))
        # Pad to target size
        pad_x = target_w - new_w
        pad_y = target_h - new_h
        top = pad_y // 2
        bottom = pad_y - top
        left = pad_x // 2
        right = pad_x - left
        image_padded = cv2.copyMakeBorder(image_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
        # Adjust keypoints to new scale and padding
        keypoints[:, 0] = keypoints[:, 0] * scale + left
        keypoints[:, 1] = keypoints[:, 1] * scale + top

        # Data augmentation (flip, rotate) if training
        if self.is_train:
            image_padded, _, keypoints, visible = transforms.random_flip(image_padded, None, keypoints, visible)
            image_padded, _, keypoints, visible = transforms.random_rotate_and_scale(image_padded, None, keypoints, visible)
        # Normalize image
        image_norm = transforms.normalize_image(image_padded)
        # Generate target heatmaps (output size will be smaller than input)
        heatmaps = transforms.generate_heatmaps(keypoints, visible, output_size=(target_h//4, target_w//4))
        # To tensor
        image_tensor = transforms.to_tensor(image_norm).float()
        heatmaps_tensor = transforms.to_tensor(heatmaps).float()
        return image_tensor, heatmaps_tensor
