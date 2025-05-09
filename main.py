import os
import torch
import wandb
from torch.utils.data import DataLoader
# 导入模型、损失函数、训练工具和评估函数
from models.Multi_Pose import MultiPoseNet  # 假设模型定义在 models.pose_model
from utils.loss import PoseLoss
from utils.train_utils import train_one_epoch, ModelEMA
from utils.evaluate import evaluate
from utils.scheduler import build_onecycle_scheduler
# 导入数据集类 (假定实现了 COCO 骨骼数据集)
from utils.coco import COCOPoseDataset

if __name__ == "__main__":
    # 配置超参数和训练设置
    epochs = 50
    batch_size = 16
    learning_rate = 5e-4
    use_amp = True          # 是否使用混合精度训练
    use_ema = True          # 是否使用EMA滑动平均模型
    grad_clip = 5.0         # 梯度裁剪阈值 (None表示不裁剪)
    ohkm_k = 8              # 在线困难关键点挖掘(OHKM)选取的关键点数
    struct_weight = 0.0     # 骨架结构约束损失权重 (0表示不使用结构损失)
    n_vis = 3               # 每轮随机可视化验证集样本数量

    # 初始化 Weights & Biases 日志
    wandb.init(project="PoseEstimation", config={
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "use_amp": use_amp,
        "use_ema": use_ema,
        "grad_clip": grad_clip,
        "ohkm_k": ohkm_k,
        "struct_weight": struct_weight,
        "n_vis": n_vis
    })
    config = wandb.config

    # 构建训练和验证数据集与数据加载器
    # 请根据实际数据路径和实现调整参数
    train_dataset = COCOPoseDataset(
        root="data/coco",
        ann_file="annotations/person_keypoints_train2017.json",
        img_folder="train2017"
    )
    val_dataset = COCOPoseDataset(
        root="data/coco",
        ann_file="annotations/person_keypoints_val2017.json",
        img_folder="val2017"
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    # 初始化模型、损失函数、优化器等
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiPoseNet(backbone="mobilenetv2", num_keypoints=17, refine=True)  # 假设PoseModel使用MobileNetV2骨干
    model.to(device)
    criterion = PoseLoss(ohkm_k=config.ohkm_k, struct_weight=config.struct_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # 使用OneCycleLR学习率调度替代StepLR
    scheduler = build_onecycle_scheduler(optimizer, train_loader, config.epochs, config.learning_rate)
    # 初始化EMA模型
    ema = ModelEMA(model, decay=0.999) if config.use_ema else None

    # 监视模型参数和梯度（WandB）
    wandb.watch(model, log="all", log_freq=100)

    best_ap = 0.0
    # 训练循环
    for epoch in range(config.epochs):
        # 单个epoch训练
        avg_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            use_amp=config.use_amp, grad_clip=config.grad_clip,
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
            torch.save(model.state_dict(), "best_model.pth")
            if ema is not None:
                torch.save(ema.ema_model.state_dict(), "best_model_ema.pth")
        print(f"Epoch {epoch+1}/{config.epochs} - Loss: {avg_loss:.4f} - mAP: {mean_ap:.4f} - AP50: {ap50:.4f}")

    # 保存最后一轮模型权重
    torch.save(model.state_dict(), "final_model.pth")
    if ema is not None:
        torch.save(ema.ema_model.state_dict(), "final_model_ema.pth")
    wandb.finish()
