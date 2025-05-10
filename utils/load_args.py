import argparse
import yaml

def get_config():
    parser = argparse.ArgumentParser(description='Train MultiPoseNet with config file')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg