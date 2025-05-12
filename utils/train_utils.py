import torch
from tqdm import tqdm

def train_one_epoch(epoch, model, loader, criterion, optimizer, scheduler, coord_weight, device):
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
    epoch_loss_detail = {
        'loss_heatmap': 0.0,
        'loss_coord': 0.0,
        'coord_weight': 0.0
    }

    # Iterate over training set
    for i, batch in tqdm(enumerate(loader), desc="Training", total=len(loader), unit="batch"):
        # Unpack batch to images, targets (and mask if provided)
        images, targets_heatmap, targets_keypoints, mask = batch
        if i==0:  print(targets_heatmap)

        images = images.to(device)
        targets_heatmap = targets_heatmap.to(device)
        targets_keypoints = targets_keypoints.to(device)
        if mask is not None:
            mask = mask.to(device)
        optimizer.zero_grad()
        # Forward pass
        outputs = model(images)
        # Model output can be tuple (init, refine) or single
        heatmaps_init, heatmaps_refine, keypoints = outputs
        # Compute loss (apply mask if available)
        loss, loss_detail= criterion(heatmaps_refine, targets_heatmap, keypoints, targets_keypoints, mask, coord_weight)
        # Backpropagation and optimizer step
        loss.backward()
        optimizer.step()
        # OneCycleLR step (update learning rate)
        if scheduler:
            scheduler.step()
        epoch_loss += loss.item() * images.size(0)
        epoch_loss_detail['loss_heatmap'] += loss_detail['loss_heatmap']
        epoch_loss_detail['loss_coord'] += loss_detail['loss_coord']
        epoch_loss_detail['coord_weight'] = loss_detail['coord_weight']


    return epoch_loss, epoch_loss_detail
