import wandb


class WandbLogger:
    def __init__(self, _project, _entity, config=None):
        """
        初始化 wandb 运行
        :param project_name: wandb 项目名称
        :param config: 配置字典（超参数等）
        """
        wandb.init(
            project=_project,
            entity=_entity,
            config=config
        )

    def log(self, metrics_dict, step=None):
        """
        记录指标
        :param metrics_dict: 包含标量指标的字典
        :param step: 当前步数（如 epoch 编号）
        """
        wandb.log(metrics_dict, step=step)

    def finish(self):
        """结束 wandb 运行。"""
        wandb.finish()
