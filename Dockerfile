# Dockerfile: PyTorch 训练环境
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
LABEL authors="ryen"

# 禁用交互式安装
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# 安装系统依赖（opencv、ffmpeg 等常用库）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender-dev \
      libgl1 \
      ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# 升级 pip 并安装 Python 依赖
RUN pip install --upgrade pip

# 复制并安装项目依赖
WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r requirements.txt

# 复制项目代码
COPY . /workspace

# 默认训练入口（可按需调整）
CMD ["python", "train.py", "--data_dir", "run/data", "--output_dir", "run"]
