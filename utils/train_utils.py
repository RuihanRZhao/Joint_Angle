import torch
from torch.amp import autocast, GradScaler
import torch.nn.utils as torch_utils
import copy
import wandb
from tqdm import tqdm

def train_one_epoch(model, loader, criterion, optimizer, device, use_amp=False, grad_clip=None, ema=None, scheduler=None):
    """
    单个训练epoch过程，可选AMP混合精度、梯度裁剪和EMA更新。
    参数:
      model: 待训练模型(nn.Module)。
      loader: 训练数据集DataLoader。
      criterion: 损失函数，可处理模型输出和真值。
      optimizer: 优化器实例。
      device: 训练设备。
      use_amp: 是否使用自动混合精度 (bool)。
      grad_clip: 梯度裁剪阈值 (float 或 None，不裁剪)。
      ema: EMA权重更新器实例 (ModelEMA或类似对象，提供update(model)方法)。
      scheduler: 学习率调度器 (如OneCycleLR)，若提供则在每个batch后调用 step()。
    返回:
      avg_loss: 当前epoch的平均损失值。
    """
    model.train()
    scaler = GradScaler('cuda', enabled=use_amp)  # GradScaler在use_amp=True时启用
    epoch_loss = 0.0
    num_samples = 0
    # 遍历训练集
    for images, heatmaps, pafs, heatmap_weights, paf_weights, joint_coords, n_person in tqdm(loader, desc="Training", unit="batch", leave=True):
        imgs = images.to(device, non_blocking=True)
        heatmaps = heatmaps.to(device, non_blocking=True)
        pafs = pafs.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        # 前向传播（混合精度上下文）
        with autocast('cuda', enabled=use_amp):

            outputs = model(imgs)

            targets = {
                "heatmap": heatmaps,
                "paf": pafs,
                "heatmap_weight": heatmap_weights,
                "paf_weight": paf_weights,
                "joint_coords": joint_coords,
            }

            loss = criterion(outputs, targets)
        # 反向传播
        if use_amp:
            # 混合精度：将loss缩放后反传，避免精度损失
            scaler.scale(loss).backward()
            # 如使用梯度裁剪，先反缩放再裁剪FP32梯度
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch_utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch_utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
        # 更新EMA权重（如果有提供）
        if ema is not None:
            ema.update(model)
        # 每个batch后更新学习率（OneCycleLR等需按iteration更新）
        if scheduler is not None:
            scheduler.step()
        # 累积损失和样本数
        batch_size = imgs.size(0)
        epoch_loss += loss.item() * batch_size
        num_samples += batch_size
    avg_loss = epoch_loss / num_samples if num_samples > 0 else 0.0
    return avg_loss

class ModelEMA:
    """
    模型EMA (指数滑动平均) 更新器。
    保存模型的EMA副本，并在每次训练迭代后更新EMA权重，以获得更平滑的模型参数。
    """
    def __init__(self, model, decay=0.999):
        """
        参数:
          model: 原始模型 (将深拷贝用于初始化EMA模型)。
          decay: EMA衰减率，越接近1表示EMA模型保留越多之前的状态。
        """
        # 深拷贝模型作为EMA模型，并设置为eval模式
        self.ema_model = copy.deepcopy(model).eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)
        self.decay = decay

    def update(self, model):
        """
        使用当前模型参数更新EMA模型。
        公式: ema_param = decay * ema_param + (1 - decay) * current_param
        """
        with torch.no_grad():
            model_state = model.state_dict()
            ema_state = self.ema_model.state_dict()
            for key, current_val in model_state.items():
                if key in ema_state:
                    ema_val = ema_state[key]
                    # 仅对浮点型参数计算EMA，非浮点(如缓冲区)直接复制
                    if current_val.dtype.is_floating_point:
                        ema_state[key].copy_(ema_val * self.decay + current_val * (1 - self.decay))
                    else:
                        ema_state[key] = current_val
