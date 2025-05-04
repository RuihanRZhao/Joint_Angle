import argparse


def arg_test():
    parser = argparse.ArgumentParser()
    # 数据参数
    parser.add_argument('--data_dir', default='run/data')
    parser.add_argument('--output_dir', default='run')
    parser.add_argument('--max_samples', type=int, default=20)

    # 训练参数
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min_delta', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--use_fp16', action='store_true', default=True)
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--val_viz_num', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)

    # 分布式参数
    parser.add_argument('--dist', action='store_true', default=True)
    parser.add_argument('--local_rank', type=int, default=0)

    # WandB参数
    parser.add_argument('--entity', default='joint_angle')
    parser.add_argument('--project_name', default='_model')

    return parser.parse_args()


def arg_real():
    parser = argparse.ArgumentParser()
    # 数据参数
    parser.add_argument('--data_dir', default='run/data')
    parser.add_argument('--output_dir', default='run')
    parser.add_argument('--max_samples', type=int, default=None)

    # 训练参数
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--min_delta', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--use_fp16', action='store_true', default=True)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--val_viz_num', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=24)

    # 分布式参数
    parser.add_argument('--dist', action='store_true', default=True)
    parser.add_argument('--local_rank', type=int, default=0)

    # WandB参数
    parser.add_argument('--entity', default='joint_angle')
    parser.add_argument('--project_name', default='_model')

    return parser.parse_args()