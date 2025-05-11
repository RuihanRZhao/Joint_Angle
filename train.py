import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

import wandb

from networks.Joint_Pose import JointPoseNet
from utils.network_utils import PoseEstimationLoss

from utils.dataset_util.coco import COCOPoseDataset
from utils import get_config, train_one_epoch, evaluate


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


def coord_weight_scheduler(epoch: int,
                           ramp_epochs: int = 60,
                           power: float = 3.0) -> float:
    """
    epoch: 当前 epoch（从 0 开始计数）
    ramp_epochs: 预计达到 weight=1 的 epoch 数（例如 60）
    power: 多项式指数，>1 时前期更缓慢，后期更陡峭
    返回值: 当前 epoch 下 coord_weight ∈ [0,1]
    """
    t = epoch / ramp_epochs
    if t >= 1.0:
        return 1.0
    return t ** power



def main():
    config = get_config()

    wandb.init(project="Joint_Pose", config=config)

    # Parse input size
    input_size = (config['input_w'], config['input_h'])
    # Prepare output directory
    os.makedirs(config['checkpoint_root'], exist_ok=True)

    train_dataset = COCOPoseDataset(
        root=config['data_root'],
        ann_file="annotations/single_person_keypoints_train.json",
        img_dir="train",
        input_size=input_size,
        return_meta=False
    )
    val_dataset = COCOPoseDataset(
        root=config['data_root'],
        ann_file="annotations/single_person_keypoints_val.json",
        img_dir="val",
        input_size=input_size,
        return_meta=True
    )
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                              num_workers=config['num_workers_train'], pin_memory=config['pin_memory'], prefetch_factor=config['prefetch_factor'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False,
                            num_workers=config['num_workers_val'], pin_memory=config['pin_memory'], prefetch_factor=config['prefetch_factor'])

    # Model, criterion, optimizer, scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"============ CONFIG ============")
    for key, data in config.items():
        print(f"  {key:15} = {data}")

    print(f"  {'train_sample':15} = {len(train_dataset)}")
    print(f"  {'eval_sample':15} = {len(val_dataset)}")
    print(f"  {'device':15} = {device}")
    print(f"================================")
    model = JointPoseNet(num_joints=17)  # COCO has 17 joints

    model = model.to(device)
    criterion = PoseEstimationLoss()

    # Initialize optimizer (set initial LR lower, OneCycle will adjust)
    initial_lr = config['learning_rate'] / config['div_factor']
    optimizer = AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4,
                      betas=(0.9, 0.999))  # div_factor default is 25

    # Define OneCycleLR scheduler
    scheduler = OneCycleLR(optimizer, max_lr=config['learning_rate'], epochs=config['epochs'],
                           steps_per_epoch=len(train_loader), pct_start=config['warmup_pct'],
                           div_factor=config['div_factor'], final_div_factor=10000.0, anneal_strategy='cos')

    start_epoch = 0
    best_ap = float('-inf')
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
        coord_weight = coord_weight_scheduler(epoch+1)

        total_loss, loss_detail = train_one_epoch(epoch, model, train_loader, criterion, optimizer, scheduler, coord_weight, device)

        mean_ap, ap50, vis_images = evaluate(
            model, val_loader,
            "annotations/single_person_keypoints_val.json",
            os.path.join(config['data_root'], "val"),
            config['input_w'], config['input_h'],
            n_viz=config['n_viz']
        )
        print(f"Epoch {epoch + 1}: Total Loss: {total_loss:.6f} | Heat Loss{loss_detail['loss_heatmap']:.6f} | Keypoints Loss: {loss_detail['loss_coord']:.6f} | mAP: {mean_ap:.6f} | AP50: {ap50:.6f} | LR: {optimizer.param_groups[0]['lr']}")

        # Update best model
        if ap50 > best_ap:
            best_ap = ap50
            best_path = os.path.join(config['checkpoint_root'], f"best_model_{epoch + 1}.pth")
            save_checkpoint({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_AP': best_ap
            }, best_path)
            print(f"Saved new best model with loss {best_ap:.8f}")

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": total_loss,
            "train/heat_loss": loss_detail['loss_heatmap'],
            "train/kps_loss": loss_detail['loss_coord'],
            "val/mean_AP": mean_ap,
            "val/AP50": ap50,
            "train/learning_rate": optimizer.param_groups[0]["lr"],
            "train/kps_loss_weight": loss_detail['coord_weight'],
            "examples": vis_images  # 可视化预测结果
        })


    # Save final model
    final_path = os.path.join(config['checkpoint_root'], "final_model.pth")
    save_checkpoint({
        'epoch': config['epochs'],
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_AP': best_ap
    }, final_path)
    print(f"\nTraining completed. Final model saved at {final_path}")


if __name__ == "__main__":
    main()
