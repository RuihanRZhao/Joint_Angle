from tqdm import tqdm
import torch
from torch.cuda.amp import autocast, GradScaler
import torch.nn.utils as nn_utils

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler, ema_model=None, clip_grad=1.0):
    model.train()
    total_loss = 0.0
    total_x_loss = 0.0
    total_y_loss = 0.0
    count = 0

    for batch in tqdm(train_loader, desc=f'Training Epoch {epoch}', unit='batch'):
        images, target_x, target_y, mask = batch
        images = images.to(device)
        target_x = target_x.to(device)
        target_y = target_y.to(device)
        mask = mask.to(device)

        target_index_x = target_x.argmax(dim=2)
        target_index_y = target_y.argmax(dim=2)

        optimizer.zero_grad()
        with autocast():
            pred_x, pred_y = model(images)
            loss_dict = criterion(pred_x, pred_y, target_index_x, target_index_y, mask)
            loss = loss_dict['total_loss']

        scaler.scale(loss).backward()

        # 梯度裁剪
        scaler.unscale_(optimizer)
        nn_utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

        scaler.step(optimizer)
        scaler.update()

        # EMA 更新
        if ema_model is not None:
            update_ema(model, ema_model)

        total_loss += loss.item() * images.size(0)
        total_x_loss += loss_dict['x_loss'].item() * images.size(0)
        total_y_loss += loss_dict['y_loss'].item() * images.size(0)
        count += images.size(0)

    avg_total = total_loss / count
    avg_x = total_x_loss / count
    avg_y = total_y_loss / count
    lr = optimizer.param_groups[0]['lr']

    return {
        'epoch': epoch,
        'total_loss': avg_total,
        'x_loss': avg_x,
        'y_loss': avg_y,
        'lr': lr
    }


def update_ema(model, ema_model, decay=0.999):
    with torch.no_grad():
        for param, ema_param in zip(model.parameters(), ema_model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)
