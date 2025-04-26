import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data paths (placeholders, to be adjusted by user)
COCO_IMAGES_DIR = "/path/to/COCO/images/"        # e.g., ".../train2017/"
COCO_ANN_FILE = "/path/to/COCO/annotations.json" # e.g., instances_train2017.json or person_keypoints_train2017.json
MPII_IMAGES_DIR = "/path/to/MPII/images/"
MPII_ANN_FILE = "/path/to/MPII/annotations.json"

# Training hyperparameters
NUM_EPOCHS_SEG = 50      # Segmentation training epochs
NUM_EPOCHS_POSE = 200    # Pose training epochs
BATCH_SIZE_SEG = 8       # Batch size for segmentation (images)
BATCH_SIZE_POSE = 16     # Batch size for pose (person crops)
LEARNING_RATE_SEG = 1e-3
LEARNING_RATE_POSE = 1e-4
LR_STEP_SIZE = 50        # StepLR drop every 50 epochs (for example)
LR_GAMMA = 0.1           # multiply LR by 0.1 at each step

# Dataset parameters
SEG_IMAGE_SIZE = 512     # Images are resized/cropped to 512x512 for segmentation
POSE_CROP_SIZE = (256, 192)  # Width x Height for pose model input crop (common input size for pose)
NUM_KEYPOINTS_COCO = 17  # COCO has 17 keypoints
NUM_KEYPOINTS_MPII = 16  # MPII has 16 keypoints

# Data augmentation settings
FLIP_PROB = 0.5          # 50% chance to horizontal flip
ROTATION_FACTOR = 40     # Random rotation +/- 40 degrees (for pose)
SCALE_FACTOR = 0.3       # Random scaling +/- 30% (for pose)
# (We will define specific transform functions in utils/transforms.py)

# WandB (Weights & Biases) configuration (placeholder project/entity)
WANDB_PROJECT = "human-pose-segmentation"
WANDB_ENTITY = "your-entity-name"  # replace with your WandB username or team
