from torch.optim.lr_scheduler import OneCycleLR

def build_onecycle_scheduler(optimizer, train_loader, epochs, base_lr):
    """
    构建OneCycleLR学习率调度器。
    参数:
        optimizer: 优化器实例 (如Adam, SGD)。
        train_loader: 训练集的DataLoader，用于确定每epoch的迭代数。
        epochs: 总训练epoch数。
        base_lr: 基础学习率（可设为初始lr或比其略高的max_lr）。
    返回:
        scheduler: 配置好的OneCycleLR调度器对象。
    """
    steps_per_epoch = len(train_loader)
    # OneCycleLR参数设置:
    # max_lr 设置为 base_lr（假定传入的base_lr即我们希望的一周期内的峰值学习率）
    # pct_start 设为30%，即前30%训练过程学习率上升，其余70%下降
    # anneal_strategy 使用余弦退火，使下降过程平滑
    # div_factor 将初始lr设置为 max_lr/div_factor
    # final_div_factor 将最后的lr设置为 max_lr/final_div_factor
    scheduler = OneCycleLR(
        optimizer,
        max_lr=base_lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1e4
    )
    return scheduler

# 使用示例：
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# scheduler = build_onecycle_scheduler(optimizer, train_loader, epochs=50, base_lr=1e-3)
# 在训练循环内，每个batch优化器更新后调用 scheduler.step()
