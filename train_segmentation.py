import wandb
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from config import *
from models.segmentation_model import UNetSegmentationModel
from datasets.coco import COCOSegmentationDataset
from utils.training import train_segmentation

# Initialize wandb run
wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, config={
    "stage": "segmentation",
    "epochs": NUM_EPOCHS_SEG,
    "batch_size": BATCH_SIZE_SEG,
    "learning_rate": LEARNING_RATE_SEG,
    "image_size": SEG_IMAGE_SIZE,
    "optimizer": "Adam",
    "scheduler": "StepLR",
})

# Prepare dataset and dataloader
train_dataset = COCOSegmentationDataset(COCO_IMAGES_DIR + "train2017", COCO_ANN_FILE.replace(".json", "_train2017.json"),
                                        image_size=SEG_IMAGE_SIZE, transform=True)
val_dataset = COCOSegmentationDataset(COCO_IMAGES_DIR + "val2017", COCO_ANN_FILE.replace(".json", "_val2017.json"),
                                      image_size=SEG_IMAGE_SIZE, transform=False)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_SEG, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_SEG, shuffle=False, num_workers=2, pin_memory=True)

# Initialize model, optimizer, scheduler
model = UNetSegmentationModel(num_classes=1, encoder_name="resnet34").to(DEVICE)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE_SEG)
scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

# Train the model
train_segmentation(model, train_loader, val_loader, optimizer, scheduler, num_epochs=NUM_EPOCHS_SEG)

# Mark the run as finished (optional)
wandb.finish()
