import torch
from torch.cuda.amp import autocast, GradScaler
import torch.nn.utils as torch_utils
import copy
import wandb

def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, use_amp=False, grad_clip=None, ema=None):
    """
    单个训练epoch过程，可选AMP混合精度、梯度裁剪和EMA更新。
    参数:
        model: 待训练模型(nn.Module)。
        loader: 训练数据集DataLoader。
        criterion: 损失函数，可处理模型输出和真值。
        optimizer: 优化器。
        device: 训练设备。
        use_amp: 是否使用自动混合精度 (bool)。
        grad_clip: 梯度裁剪阈值 (float 或 None，不裁剪)。
        ema: EMA权重更新器实例 (ModelEMA或类似对象，提供update(model)方法)。
    返回:
        avg_loss: 当前epoch的平均损失。
    """
    model.train()
    scaler = GradScaler(enabled=use_amp)  # GradScaler在use_amp=True时启用
    epoch_loss = 0.0
    num_samples = 0

    # 遍历训练集
    for imgs, heatmaps, pafs in loader:
        imgs = imgs.to(device, non_blocking=True)
        heatmaps = heatmaps.to(device, non_blocking=True)
        pafs = pafs.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        # 前向传播（混合精度上下文）
        with autocast(enabled=use_amp):
            outputs = model(imgs)
            loss = criterion(outputs, (heatmaps, pafs))
        # 反向传播
        if use_amp:
            # 混合精度：将loss缩放后反传，避免精度损失
            scaler.scale(loss).backward()
        else:
            loss.backward()
        # 可选：梯度裁剪（在未更新优化器权重前进行）
        if grad_clip is not None:
            if use_amp:
                # unscale将梯度转换回FP32以便裁剪
                scaler.unscale_(optimizer)
            torch_utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        # 优化器更新参数
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        # 更新EMA权重（如果有提供）
        if ema is not None:
            ema.update(model)
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
            decay: EMA衰减率，越接近1表示更新时EMA模型保留更多先前状态。
        """
        self.ema_model = copy.deepcopy(model).eval()  # 深拷贝模型作为EMA模型
        for param in self.ema_model.parameters():
            param.requires_grad_(False)  # EMA模型不需要梯度
        self.decay = decay

    def update(self, model):
        """
        使用当前模型参数更新EMA模型。
        公式: ema_param = decay * ema_param + (1 - decay) * current_param
        """
        with torch.no_grad():
            msd = model.state_dict()       # 当前模型参数
            ema_sd = self.ema_model.state_dict()  # EMA模型参数
            for key, current_val in msd.items():
                # 如果是浮点型参数，则计算EMA; 非浮点参数如buffers直接赋值
                if key in ema_sd:
                    ema_val = ema_sd[key]
                    if current_val.dtype.is_floating_point:
                        ema_sd[key].copy_(ema_val * self.decay + current_val * (1.0 - self.decay))
                    else:
                        ema_sd[key].copy_(current_val)

    def state_dict(self):
        """获取EMA模型的参数字典，用于保存检查点。"""
        return self.ema_model.state_dict()

# 用法示例：
# model = MultiPoseNet(...).to(device)
# ema_updater = ModelEMA(model, decay=0.999)  # 初始化EMA
# for epoch in range(epochs):
#     train_one_epoch(model, train_loader, criterion, optimizer, device, use_amp=True, grad_clip=5.0, ema=ema_updater)
#     # ... 验证 ...
# # 训练结束后，可使用 ema_updater.ema_model 进行评估或保存
#