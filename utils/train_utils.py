import torch
from torch.cuda.amp import autocast, GradScaler
import torch.nn.utils as torch_utils
import copy
import wandb
from tqdm import tqdm

def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    """
    单个训练epoch过程，可选AMP混合精度、梯度裁剪和EMA更新。
    参数:
      model: 待训练模型(nn.Module)。
      loader: 训练数据集DataLoader。
      criterion: 损失函数，可处理模型输出和真值。
      optimizer: 优化器实例。
      device: 训练设备。
      use_amp: 是否使用自动混合精度 (bool)。
      grad_clip: 渐变裁剪阈值 (float 或 None，不裁剪)。
      ema: EMA权重更新器实例 (ModelEMA或类似对象，提供update(model)方法)。
      scheduler: 学习率调度器 (如OneCycleLR)，若提供则在每个batch后调用 step()。
    返回:
      avg_loss: 当前epoch的平均损失值。
    """
    model.train()
    epoch_loss = 0.0
    # Iterate over training set
    for i, batch in tqdm(enumerate(loader), desc="Training", total=len(loader), unit="batch"):
        # Unpack batch to images, targets (and mask if provided)
        if isinstance(batch, (tuple, list)) and len(batch) == 3:
            images, targets, mask = batch
        else:
            images, targets = batch
            mask = None
        images = images.to(device)
        targets = targets.to(device)
        if mask is not None:
            mask = mask.to(device)
        optimizer.zero_grad()
        # Forward pass
        outputs = model(images)
        # Model output can be tuple (init, refine) or single
        if isinstance(outputs, (tuple, list)):
            heatmaps_init, heatmaps_refine = outputs
        else:
            heatmaps_init, heatmaps_refine = outputs, None
        # Compute loss (apply mask if available)
        loss_init = criterion(heatmaps_init, targets, mask) if mask is not None else criterion(heatmaps_init, targets)
        loss_refine = criterion(heatmaps_refine, targets, mask) if heatmaps_refine is not None else 0
        loss = loss_init + (loss_refine if isinstance(loss_refine, torch.Tensor) else 0)
        # Backpropagation and optimizer step
        loss.backward()
        optimizer.step()
        # OneCycleLR step (update learning rate)
        if scheduler:
            scheduler.step()
        epoch_loss += loss.item() * images.size(0)

    return epoch_loss
