#!/bin/bash

echo "准备启动MultiSpecVision Web应用..."

# 检查预训练模型是否存在
if [ ! -f "checkpoints/swin_tiny_patch4_window7_224.pth" ]; then
    echo "正在下载预训练模型权重..."
    mkdir -p checkpoints
    wget -P checkpoints https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
fi

echo "启动MultiSpecVision Web应用..."
python app.py