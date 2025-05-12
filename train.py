import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

import wandb

from networks.Joint_Pose import JointPoseNet
from utils.network_utils import SimCCLoss

from utils.dataset_util.coco import COCOPoseDataset
from utils import get_config, train_one_epoch, evaluate


def save_checkpoint(state, path):
    torch.save(state, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    start_epoch = checkpoint.get('epoch', 0)
    best_ap = checkpoint.get('best_AP', float('-inf'))
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    print(f"Resumed training from epoch {start_epoch}.")
    return start_epoch, best_ap

def coord_weight_scheduler(epoch, ramp_epochs=10):
    return 1.0 if epoch >= ramp_epochs else epoch / ramp_epochs

if __name__ == "__main__":
    config = get_config()
    wandb.init(project="JointPose_SimCC", config=config)

    input_size = (config['input_w'], config['input_h'])
    os.makedirs(config['checkpoint_root'], exist_ok=True)

    train_dataset = COCOPoseDataset(
        root=config['data_root'], ann_file="annotations/single_person_keypoints_train.json",
        img_dir="train", input_size=input_size, return_meta=False, max_samples=config['max_samples_train'], bins=config['bins']
    )
    val_dataset = COCOPoseDataset(
        root=config['data_root'], ann_file="annotations/single_person_keypoints_val.json",
        img_dir="val", input_size=input_size, return_meta=True, max_samples=config['max_samples_val'], bins=config['bins']
    )
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                              num_workers=config['num_workers_train'], pin_memory=config['pin_memory'])
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=config['num_workers_val'], pin_memory=config['pin_memory'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = JointPoseNet(num_joints=17, input_size=input_size, bins=config['bins']).to(device)
    criterion = SimCCLoss()
    optimizer = AdamW(model.parameters(), lr=config['learning_rate']/config['div_factor'], weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=config['learning_rate'], epochs=config['epochs'],
                           steps_per_epoch=len(train_loader), pct_start=config['warmup_pct'],
                           div_factor=config['div_factor'], final_div_factor=10000.0)

    start_epoch = 0
    best_ap = float('-inf')
    if config['resume']:
        resume_path = os.path.join(config['checkpoint_root'], f"epoch_{config['resume_id']}.pth")
        if os.path.isfile(resume_path):
            start_epoch, best_ap = load_checkpoint(resume_path, model, optimizer, scheduler)

    wandb.watch(model, log="all", log_freq=100)

    for epoch in range(start_epoch, config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        metrics = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch)

        mAP, AP50, vis_images = evaluate(model, val_loader, device, input_size, config['bins'], n_viz=config['n_viz'])

        print(f"Ep {epoch+1}: Total Loss: {metrics['total_loss']:.8f} | X: {metrics['x_loss']:.8f} | Y: {metrics['y_loss']:.8f} | mAP: {mAP:.12f} | AP50: {AP50:.12f}")

        # Save best
        if AP50 > best_ap:
            best_ap = AP50
            save_path = os.path.join(config['checkpoint_root'], f"best_model_{epoch+1}.pth")
            save_checkpoint({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_AP': best_ap
            }, save_path)

        # Log
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": metrics['total_loss'],
            "train/heat_loss": metrics['x_loss'],
            "train/kps_loss": metrics['y_loss'],
            "train/learning_rate": metrics['lr'],
            "val/mean_AP": mAP,
            "val/AP50": AP50,
            "examples": vis_images
        })

    # Save final
    final_path = os.path.join(config['checkpoint_root'], "final_model.pth")
    save_checkpoint({
        'epoch': config['epochs'],
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_AP': best_ap
    }, final_path)
    print(f"\nTraining completed. Final model saved at {final_path}")



