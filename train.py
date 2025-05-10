import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

import wandb
from tqdm import tqdm, trange

from networks.Joint_Pose import JointPoseNet
from network_utils import HeatmapMSELoss

from dataset_util.coco import COCOPoseDataset
from utils import get_config, train_one_epoch


def save_checkpoint(state, path):
    """Utility to save checkpoint dictionary (model, optimizer, etc.) to the given path."""
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """Utility to load model (and optimizer/scheduler) state from checkpoint file."""
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    start_epoch = checkpoint.get('epoch', 0)
    best_loss = checkpoint.get('best_loss', float('inf'))
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    print(f"Resumed training from epoch {start_epoch}.")
    return start_epoch, best_loss


def main():
    config = get_config()

    wandb.init(project="Multi_Pose", config=config)

    # Parse input size
    input_size = (config['input_w'], config['input_h'])
    # Prepare output directory
    os.makedirs(config['checkpoint_root'], exist_ok=True)

    train_dataset = COCOPoseDataset(
        root=config['data_root'],
        ann_file="annotations/person_keypoints_train2017.json",
        img_dir="train2017",
        input_size=input_size,
    )
    val_dataset = COCOPoseDataset(
        root=config['data_root'],
        ann_file="annotations/person_keypoints_val2017.json",
        img_dir="val2017",
        input_size=input_size,
    )
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers_train'], prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers_val'], prefetch_factor=2)

    # Model, criterion, optimizer, scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = JointPoseNet(num_joints=17)  # COCO has 17 joints
    
    # Note: If Joint_Pose.py forward uses variable 'heatmap1', ensure it is corrected to 'heatmap_init'
    model = model.to(device)
    criterion = HeatmapMSELoss()
    
    # Initialize optimizer (set initial LR lower, OneCycle will adjust)

    initial_lr = config['learning_rate']/config['div_factor']
    optimizer = AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4, betas=(0.9, 0.999))  # div_factor default is 25
    
    # Define OneCycleLR scheduler
    total_steps = config['epochs'] * len(train_loader)
    scheduler = OneCycleLR(optimizer, max_lr=config['learning_rate'], epochs=config['epochs'],
                           steps_per_epoch=len(train_loader), pct_start=config['warmup_pct'],
                           div_factor=config['div_factor'], final_div_factor=10000.0, anneal_strategy='cos')

    start_epoch = 0
    best_loss = float('inf')
    # Resume from checkpoint if specified
    if config['resume']:
        resume_model_path = os.path.join(config['checkpoint_root'], f"epoch_{config['resume_id']}.pth")
        if os.path.isfile(resume_model_path):
            # Initialize model & optimizer before loading scheduler state
            start_epoch, best_loss = load_checkpoint(resume_model_path, model, optimizer, scheduler)
            # If resuming, adjust total_steps in scheduler if needed (assuming same epochs count as originally planned)
        else:
            print(f"Resume checkpoint not found: {resume_model_path}")

    wandb.watch(model, log="all", log_freq=100)
    # Training loop
    for epoch in range(start_epoch, config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        print(f"Epoch {epoch + 1} finished. Average Loss: {avg_loss:.4f}")


        # Update best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(config['checkpoint_root'], f"best_model_{epoch+1}.pth")
            save_checkpoint({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_loss': best_loss
            }, best_path)
            print(f"Saved new best model with loss {best_loss:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_mean_AP": mean_ap,
            "val_AP50": ap50,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "examples": vis_images  # 可视化预测结果
        })



    # Save final model
    final_path = os.path.join(config['checkpoint_root'], "final_model.pth")
    save_checkpoint({
        'epoch': config['epochs'],
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_loss': best_loss
    }, final_path)
    print(f"\nTraining completed. Final model saved at {final_path}")


if __name__ == "__main__":
    main()
