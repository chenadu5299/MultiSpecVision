FROM pytorch/pytorch:latest

WORKDIR /app

# 安装依赖
RUN apt-get update && apt-get install -y \
    wget \
    libraw-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# 复制应用代码
COPY . /app/

# 创建检查点目录
RUN mkdir -p checkpoints

# 下载预训练模型（如果不存在）
RUN if [ ! -f checkpoints/swin_tiny_patch4_window7_224.pth ]; then \
    wget -P checkpoints https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth; \
    fi

# 暴露端口
EXPOSE 5000

# 启动应用
CMD ["python", "multi_channel_app.py"] 