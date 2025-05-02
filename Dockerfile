# Dockerfile: PyTorch 训练环境，支持非交互式安装，修复 FiftyOne mongod 启动
FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime

# 设置非交互式安装环境，防止 tzdata 配置交互提示
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# 安装系统依赖（包含 libcurl4 以支持 FiftyOne 的 mongod）
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    tzdata \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libcurl4 \
 && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
RUN pip install --upgrade pip
RUN pip install --no-cache-dir \
    torchvision \
    pillow \
    numpy \
    wandb \
    fiftyone \
    opencv-python \
    pycocotools

# 设置工作目录
WORKDIR /workspace

# 复制项目代码
COPY . /workspace

# 防止 Python 输出被缓存
ENV PYTHONUNBUFFERED=1

# 入口命令：运行训练脚本
CMD ["python", "train.py", "--data_dir", "run/data", "--output_dir", "run"]
