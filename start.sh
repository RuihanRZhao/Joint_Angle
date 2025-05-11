#!/usr/bin/env bash
# 卸载现有 PyTorch、TorchVision、TorchAudio
#echo "卸载旧版 torch、torchvision、torchaudio..."
#pip3 uninstall -y torch torchvision torchaudio

# 安装与 CUDA 12.8 兼容的 PyTorch 2.7.0 及配套包
#echo "安装 torch==2.7.0、torchvision==0.22.0、torchaudio==2.7.0（CUDA 12.8 支持）..."
#pip3 install --upgrade pip
#pip3 install \
#    torch==2.7.0 \
#    torchvision==0.22.0 \
#    torchaudio==2.7.0 \
#    --index-url https://download.pytorch.org/whl/cu128


# 安装完成后，简单验证
echo "验证安装结果："
python3 - <<'EOF'
import torch, torchvision
print("PyTorch:", torch.__version__, "| CUDA:", torch.version.cuda)
print("TorchVision:", torchvision.__version__)
from torchvision.ops import nms
print("nms operator loaded:", callable(nms))
EOF


pip3 install -r requirements.txt


mkdir run/
mkdir run/data
mkdir run/single_person
mkdir run/checkpoints

python single_person_dataset.py