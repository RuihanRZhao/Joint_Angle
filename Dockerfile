# Dockerfile: PyTorch 训练环境，添加 libcurl4 依赖修复 FiftyOne mongod 启动失败
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# 安装系统依赖（包含 libcurl4 以支持 FiftyOne 的 mongod）
RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libcurl4 \
 && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
RUN pip install --upgrade pip
RUN pip install \
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
