import cv2
import numpy as np
import torch
from config import SEG_IMAGE_SIZE, POSE_CROP_SIZE, FLIP_PROB, ROTATION_FACTOR, SCALE_FACTOR

# Constants for normalization (ImageNet mean/std for ResNet backbone)
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

def normalize_image(img):
    """Normalize image tensor to zero mean and unit variance (using ImageNet stats)."""
    img = img.astype(np.float32) / 255.0
    # subtract mean and divide by std
    img -= np.array(IMG_MEAN, dtype=np.float32)
    img /= np.array(IMG_STD, dtype=np.float32)
    return img

def resize_image_and_mask(image, mask, size):
    """Resize image (and mask if provided) to the given square size."""
    h, w = image.shape[:2]
    # Resize with aspect ratio preserved and pad, or simple warp (here we do simple stretch for simplicity)
    image_resized = cv2.resize(image, (size, size))
    mask_resized = None
    if mask is not None:
        mask_resized = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
        mask_resized = (mask_resized > 0.5).astype(np.uint8)  # ensure binary
    return image_resized, mask_resized

def random_flip(image, mask=None, keypoints=None, keypoints_visible=None):
    """Random horizontal flip for image, with mask or keypoints."""
    if np.random.rand() < FLIP_PROB:
        image = np.fliplr(image).copy()
        if mask is not None:
            mask = np.fliplr(mask).copy()
        if keypoints is not None:
            # keypoints: numpy array of shape (N,2), visible flags: (N,)
            keypoints[:, 0] = image.shape[1] - 1 - keypoints[:, 0]  # flip x coordinate
            # Swap left-right joints if visible
            if keypoints_visible is not None:
                # Define pairs of indices to swap (for COCO 17 keypoints or MPII 16 keypoints)
                # Example for COCO: left_right_pairs = [(1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14), (15,16)]
                # (This maps left eye<->right eye, left ear<->right ear, left shoulder<->right shoulder, etc.)
                # For simplicity, define common pairs for both COCO and MPII:
                left_right_pairs = []
                if keypoints.shape[0] == 17:  # COCO
                    left_right_pairs = [(1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14), (15,16)]
                elif keypoints.shape[0] == 16:  # MPII
                    # MPII ordering might be: [0: right ankle,1: right knee,2: right hip,3: left hip,4: left knee,5: left ankle,
                    # 6: right wrist,7: right elbow,8: right shoulder,9: left shoulder,10: left elbow,11: left wrist,
                    # 12: neck, 13: head top, 14: (maybe not used?), 15: (maybe not used?)], or similar.
                    # For simplicity, define a plausible mapping for MPII (actual indices may differ in annotation):
                    left_right_pairs = [(0,5), (1,4), (2,3), (6,11), (7,10), (8,9)]
                # Swap the keypoint coordinates and visibility
                for (li, ri) in left_right_pairs:
                    keypoints[[li, ri]] = keypoints[[ri, li]]
                    if keypoints_visible is not None:
                        keypoints_visible[[li, ri]] = keypoints_visible[[ri, li]]
    return image, mask, keypoints, keypoints_visible

def random_rotate_and_scale(image, mask=None, keypoints=None, keypoints_visible=None):
    """Randomly rotate and scale the image (for pose augmentation), adjusting keypoints accordingly."""
    # Rotation (degrees) and scale factor
    angle = np.random.uniform(-ROTATION_FACTOR, ROTATION_FACTOR)
    scale = np.random.uniform(1 - SCALE_FACTOR, 1 + SCALE_FACTOR)
    h, w = image.shape[:2]
    center = (w/2, h/2)
    # Compose rotation+scale matrix
    M = cv2.getRotationMatrix2D(center, angle, scale)
    # Apply to image
    image_rot = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask_rot = None
    if mask is not None:
        mask_rot = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # Apply transform to keypoints
    if keypoints is not None:
        # augment keypoints (and visibility)
        ones = np.ones((keypoints.shape[0], 1))
        pts_homog = np.concatenate([keypoints, ones], axis=1)  # make (x,y,1)
        pts_rot = (M @ pts_homog.T).T  # apply affine transform
        keypoints = pts_rot[:, :2]
        # If a keypoint goes outside image after transform, we can mark it invisible
        if keypoints_visible is not None:
            for i, (x, y) in enumerate(keypoints):
                if not (0 <= x < w and 0 <= y < h):
                    keypoints_visible[i] = 0
    else:
        keypoints_visible = None
    return image_rot, mask_rot, keypoints, keypoints_visible

def generate_heatmaps(keypoints, visible, output_size, sigma=2):
    """Generate heatmap for each keypoint. output_size is (h, w) of heatmap."""
    num_kpts = keypoints.shape[0]
    h, w = output_size
    heatmaps = np.zeros((num_kpts, h, w), dtype=np.float32)
    tmp_size = sigma * 3
    # Downscale keypoint coordinates to heatmap size (assuming image was POSE_CROP_SIZE and heatmap is POSE_CROP_SIZE/4)
    scale_x = w / float(POSE_CROP_SIZE[0])
    scale_y = h / float(POSE_CROP_SIZE[1])
    for k in range(num_kpts):
        if visible is not None and visible[k] == 0:
            continue  # skip invisible keypoints
        x = keypoints[k, 0] * scale_x
        y = keypoints[k, 1] * scale_y
        if x < 0 or y < 0 or x >= w or y >= h:
            continue  # keypoint out of bounds
        # Gaussian upper-left and bottom-right corners
        ul = int(x - tmp_size), int(y - tmp_size)
        br = int(x + tmp_size + 1), int(y + tmp_size + 1)
        # Gaussian size
        size = 2 * tmp_size + 1
        # Create gaussian
        x_coords = np.arange(0, size, 1, np.float32)
        y_coords = x_coords[:, np.newaxis]
        x0 = y0 = size // 2
        gaussian = np.exp(- ((x_coords - x0)**2 + (y_coords - y0)**2) / (2 * sigma**2))
        # Compute usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], w) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], h) - ul[1]
        img_x = max(0, ul[0]), min(br[0], w)
        img_y = max(0, ul[1]), min(br[1], h)
        if g_x[0] < g_x[1] and g_y[0] < g_y[1]:
            heatmaps[k, img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
                heatmaps[k, img_y[0]:img_y[1], img_x[0]:img_x[1]],
                gaussian[g_y[0]:g_y[1], g_x[0]:g_x[1]]
            )
    return heatmaps

def to_tensor(np_array):
    """Convert numpy array to PyTorch tensor (CHW format)"""
    if np_array.ndim == 3:
        # For image: HWC to CHW
        np_array = np_array.transpose(2, 0, 1)
    return torch.from_numpy(np_array)
