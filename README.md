# Joint Angle
Human Segmentation and Pose Estimation Project

## Overview
This project implements a two-stage deep learning pipeline for human analysis:
1. **Human Segmentation** (Stage 1) using a UNet variant.
2. **Human Pose Estimation** (Stage 2) using a ResNet-based model.

We train on the COCO and MPII datasets, and integrate Weights & Biases (wandb) for experiment tracking.

## Structure
- `config.py`: Configuration and hyperparameters.
- `models/`: Model definitions.
- `datasets/`: Dataset and dataloader implementations.
- `utils/`: Utility modules for transforms and training loops.
- `train_segmentation.py`: Script to train the segmentation model.
- `train_pose.py`: Script to train the pose estimation model.
- `requirements.txt`: Python dependencies.

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Edit `config.py` to set dataset paths and wandb project details.
3. Train segmentation: `python train_segmentation.py`
4. Train pose estimation: `python train_pose.py`
5. Trained models saved as `best_segmentation_model.pth` and `best_pose_model.pth`.

