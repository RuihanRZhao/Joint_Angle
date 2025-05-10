import os
import torch
import wandb
import argparse
import yaml

from torch.utils.data import DataLoader
# 导入模型、损失函数、训练工具和评估函数
from networks.Joint_Pose import MultiPoseNet  # 假设模型定义在 networks.pose_model
from utils.loss import PoseLoss
from utils.train_utils import train_one_epoch, ModelEMA
from utils.evaluate import evaluate
from utils.scheduler import build_onecycle_scheduler
# 导入数据集类 (假定实现了 COCO 骨骼数据集)
from utils.coco import COCOPoseDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Train MultiPoseNet with config file')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

if __name__ == "__main__":
    cfg = parse_args()

    print(f"============ CONFIG ============")
    for key,data in cfg.items():
        print(f"  {key:15} = {data}")
    print(f"================================")

    # 配置超参数和训练设置
    data_root = cfg['data_root']
    batch_size = cfg['batch_size']
    learning_rate = cfg['learning_rate']
    epochs = cfg['epochs']
    img_h, img_w = cfg['img_h'], cfg['img_w']
    hm_h, hm_w = cfg['hm_h'], cfg['hm_w']
    sigma = cfg['sigma']
    ohkm_k = cfg['ohkm_k']
    n_vis = cfg['n_vis']
    num_workers_train = cfg['num_workers_train']
    num_workers_val = cfg['num_workers_val']
    grad_clip = cfg['grad_clip']
    seed = cfg['seed']
    use_amp = cfg['use_amp']
    use_ema = cfg['use_ema']
    struct_weight = cfg['struct_weight']

    # 初始化 Weights & Biases 日志
    wandb.init(project="Multi_Pose", config={
        'batch_size':  batch_size,
        'learning_rate':  learning_rate,
        'epochs':  epochs,
        'img_h':  img_h,
        'img_w':  img_w,
        'hm_h':  hm_h,
        'hm_w':  hm_w,
        'sigma':  sigma,
        'ohkm_k':  ohkm_k,
        'n_vis':  n_vis,
        'num_workers_train':  num_workers_train,
        'num_workers_val':  num_workers_val,
        'grad_clip':  grad_clip,
        'seed':  seed,
        'use_amp':  use_amp,
        'use_ema':  use_ema,
        'struct_weight':  struct_weight,
    })
    config = wandb.config

    # 构建训练和验证数据集与数据加载器
    # 请根据实际数据路径和实现调整参数
    train_dataset = COCOPoseDataset(
        root=data_root,
        ann_file="annotations/person_keypoints_train2017.json",
        img_dir="train2017",
        input_size=(img_h, img_w),
        hm_size=(hm_h, hm_w)
    )
    val_dataset = COCOPoseDataset(
        root=data_root,
        ann_file="annotations/person_keypoints_val2017.json",
        img_dir="val2017",
        input_size=(img_h, img_w),
        hm_size=(hm_h, hm_w)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers_train, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False, num_workers=num_workers_val, prefetch_factor=2)

    # 初始化模型、损失函数、优化器等
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiPoseNet(num_keypoints=17, refine=True)  # 假设PoseModel使用MobileNetV2骨干
    model.to(device)
    criterion = PoseLoss(ohkm_k=0, struct_weight=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 使用OneCycleLR学习率调度替代StepLR
    scheduler = build_onecycle_scheduler(optimizer, train_loader, epochs, learning_rate)
    # 初始化EMA模型
    ema = ModelEMA(model, decay=0.999) if use_ema else None

    # 监视模型参数和梯度（WandB）
    wandb.watch(model, log="all", log_freq=100)

    best_ap = 0.0
    # 训练循环
    for epoch in range(epochs):
        if epoch > 150:
            criterion.ohkm_k = ohkm_k

        if best_ap >= 0.3:
            criterion.struct_weight = 0.1



        # 单个epoch训练
        avg_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            use_amp=use_amp, grad_clip=grad_clip,
            ema=ema, scheduler=scheduler
        )
        # 使用EMA模型进行评估（如果启用EMA）以获得平滑的验证结果
        eval_model = ema.ema_model if ema is not None else model
        eval_model.to(device)
        mean_ap, ap50, vis_images = evaluate(eval_model, val_loader, device)
        # 记录指标到WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_mean_AP": mean_ap,
            "val_AP50": ap50,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "examples": vis_images  # 可视化预测结果
        })
        # 保存当前最佳模型权重
        if mean_ap > best_ap:
            best_ap = mean_ap
            torch.save(model.state_dict(), f"run/models/best_model_ep{epoch + 1}.pth")
            print(f"Saving best model on ep{epoch + 1}.pth")
            if ema is not None:
                torch.save(ema.ema_model.state_dict(), f"run/models/best_model_ema_ep{epoch + 1}.pth")
                print(f"Saving best ema model on ep{epoch + 1}.pth")


        print(f"Epoch {epoch+1}/{config.epochs} - Loss: {avg_loss:.12f} | mAP: {mean_ap:.12f} | AP50: {ap50:.12f}")

    # 保存最后一轮模型权重
    torch.save(model.state_dict(), "final_model.pth")
    if ema is not None:
        torch.save(ema.ema_model.state_dict(), "final_model_ema.pth")
    wandb.finish()
