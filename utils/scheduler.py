from torch.optim.lr_scheduler import OneCycleLR

def build_onecycle_scheduler(optimizer, train_loader, epochs, base_lr):
    """
    构建 OneCycleLR 学习率调度器。
    参数:
      optimizer: 优化器实例 (例如 Adam 或 SGD)。
      train_loader: 训练集的 DataLoader，用于确定每个epoch的迭代数。
      epochs: 总训练 epoch 数。
      base_lr: 基础学习率（将作为 OneCycleLR 的 max_lr）。
    返回:
      scheduler: 配置好的 OneCycleLR 调度器对象。
    """
    steps_per_epoch = len(train_loader)
    # OneCycleLR 参数配置:
    # max_lr 设置为 base_lr（假定 base_lr 为在 OneCycle 中希望的峰值学习率）
    # pct_start=0.3 表示前30%训练过程学习率上升，其余70%下降
    # anneal_strategy='cos' 采用余弦退火策略降低学习率
    # div_factor 初始学习率将设置为 max_lr/div_factor
    # final_div_factor 最终学习率将是 max_lr/final_div_factor
    scheduler = OneCycleLR(
        optimizer, max_lr=base_lr, epochs=epochs, steps_per_epoch=len(train_loader),
        pct_start=0.1,  # 将前10%训练用于warm-up
        anneal_strategy='cos',
        div_factor=10.0,  # 初始学习率 = max_lr/10
        final_div_factor=1e4  # 最终学习率 = max_lr/1e4
    )
    return scheduler
