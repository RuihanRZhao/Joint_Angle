import wandb
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from config import *
from models.pose_model import PoseEstimationModel
from datasets.coco import COCOKeypointsDataset
from datasets.mpii import MPIIPoseDataset
from utils.training import train_pose

# Initialize wandb for pose training
wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, config={
    "stage": "pose",
    "epochs": NUM_EPOCHS_POSE,
    "batch_size": BATCH_SIZE_POSE,
    "learning_rate": LEARNING_RATE_POSE,
    "optimizer": "Adam",
    "scheduler": "StepLR",
    "backbone": "ResNet50",
    "input_size": POSE_CROP_SIZE,
    "num_keypoints": NUM_KEYPOINTS_COCO
})

# Prepare dataset and loader (choose COCO or MPII or combined)
train_dataset = COCOKeypointsDataset(COCO_IMAGES_DIR + "train2017", COCO_ANN_FILE.replace(".json", "_train2017.json"), is_train=True)
val_dataset = COCOKeypointsDataset(COCO_IMAGES_DIR + "val2017", COCO_ANN_FILE.replace(".json", "_val2017.json"), is_train=False)
# If using MPII:
# train_dataset = MPIIPoseDataset(MPII_IMAGES_DIR, mpii_train_annotations_list, is_train=True)
# val_dataset = MPIIPoseDataset(MPII_IMAGES_DIR, mpii_val_annotations_list, is_train=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_POSE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_POSE, shuffle=False, num_workers=2, pin_memory=True)

# Initialize model
model = PoseEstimationModel(backbone="resnet50", num_keypoints=NUM_KEYPOINTS_COCO).to(DEVICE)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE_POSE)
scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

# (Optional) If we want to use a pre-trained segmentation model to filter images or for any reason:
# e.g., we could load the segmentation model and use it in data loading to ignore images without people, etc.
# But since our dataset builder already does that via annotations, we skip loading seg model here.

# Train pose model
train_pose(model, train_loader, val_loader, optimizer, scheduler, num_epochs=NUM_EPOCHS_POSE, num_keypoints=NUM_KEYPOINTS_COCO)

wandb.finish()
