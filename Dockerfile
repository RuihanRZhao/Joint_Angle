# Dockerfile: PyTorch 训练环境
FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime
LABEL authors="ryen"
# 安装系统依赖（视需要可添加）
RUN apt-get update && apt-get install -y git libglib2.0-0 libsm6 libxext6 libxrender-dev

# 安装 Python 依赖
RUN pip install --upgrade pip
RUN pip install \
    torchvision \
    pillow \
    numpy \
    wandb \
    fiftyone \
    opencv-python

# 设置工作目录
WORKDIR /workspace
# 复制项目代码（假设 train.py 等文件在当前目录）
COPY . /workspace

# （可选）设置环境变量，如指定 CUDA 设备等
ENV PYTHONUNBUFFERED=1

# 运行训练脚本（可根据需要修改入口命令）
CMD ["python", "train.py", "--data_dir", "run/data", "--output_dir", "run"]
