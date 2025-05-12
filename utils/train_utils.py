import torch
import torch.nn.utils as nn_utils
from torch.cuda.amp import autocast
from tqdm import tqdm

from tqdm import tqdm

def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0.0
    total_x_loss = 0.0
    total_y_loss = 0.0
    count = 0

    use_soft_label = getattr(criterion, 'use_soft', False)

    for batch in tqdm(train_loader, desc='Training', total=len(train_loader), unit='batch'):
        images, target_x, target_y, mask = batch
        images = images.to(device)
        target_x = target_x.to(device)
        target_y = target_y.to(device)
        mask = mask.to(device)

        # SimCC logits
        pred_x, pred_y = model(images)

        if not use_soft_label:
            # 若为 one-hot，需 argmax 提取 index
            target_index_x = target_x.argmax(dim=2)
            target_index_y = target_y.argmax(dim=2)
            loss_dict = criterion(pred_x, pred_y, target_index_x, target_index_y, mask)
        else:
            # 直接传 soft label tensor
            loss_dict = criterion(pred_x, pred_y, target_x, target_y, mask)

        loss = loss_dict['total_loss']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

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
