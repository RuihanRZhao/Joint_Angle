from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR


def load_optimizer(model, max_lr, total_epochs, steps_per_epoch,
                                 div_factor=25.0, pct_start=0.3):
    # 计算初始学习率并初始化 AdamW 优化器
    initial_lr = max_lr / div_factor  # 初始学习率
    optimizer = AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4, betas=(0.9, 0.999))

    # 初始化 OneCycleLR 学习率调度器
    scheduler = OneCycleLR(optimizer, max_lr=max_lr, epochs=total_epochs,
                           steps_per_epoch=steps_per_epoch, pct_start=pct_start,
                           div_factor=div_factor, final_div_factor=10000.0, anneal_strategy='cos')


def load_scheduler(model, max_lr, total_epochs, steps_per_epoch,
                                 div_factor=25.0, pct_start=0.3):
    # 计算初始学习率并初始化 AdamW 优化器
    initial_lr = max_lr / div_factor

    # 初始化 OneCycleLR 学习率调度器
    scheduler = OneCycleLR(optimizer, max_lr=max_lr, epochs=total_epochs,
                           steps_per_epoch=steps_per_epoch, pct_start=pct_start,
                           div_factor=div_factor, final_div_factor=10000.0, anneal_strategy='cos')