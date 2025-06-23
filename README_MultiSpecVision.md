# MultiSpecVision - 多光谱视觉变换器

## 项目概述

MultiSpecVision是一个基于分层视觉变换器（Hierarchical Vision Transformer）的多光谱图像处理系统，采用移位窗口（Shifted Windows）技术实现高效的图像分析和分类。

## 核心特性

### 🔥 多通道支持
- 支持3-20通道的多光谱图像处理
- 动态通道数适配
- 智能通道权重分配

### 🚀 高性能架构
- 基于Swin Transformer架构优化
- 移位窗口注意力机制
- 层次化特征提取

### 🌐 Web应用界面
- Flask Web应用支持
- 实时图像上传和处理
- 多通道图像可视化

### 🐳 容器化部署
- Docker支持
- docker-compose一键部署
- 跨平台兼容

## 项目结构

```
MultiSpecVision/
├── models/                          # 核心模型文件
│   ├── swin_transformer.py         # 标准Swin Transformer模型
│   ├── multi_channel_swin_transformer.py  # 多通道Swin Transformer
│   └── dng_processor.py            # DNG文件处理器
├── templates/                       # Web界面模板
│   ├── index.html                  # 单通道界面
│   └── multi_channel_index.html    # 多通道界面
├── checkpoints/                     # 预训练模型
│   └── swin_tiny_patch4_window7_224.pth
├── app.py                          # 主Web应用
├── multi_channel_app.py            # 多通道Web应用
├── test_inference.py               # 推理测试脚本
├── run.sh                          # 启动脚本
├── requirements.txt                # Python依赖
├── Dockerfile                      # Docker构建文件
└── docker-compose.yml             # Docker组合配置
```

## 快速开始

### 环境要求
- Python 3.8+
- PyTorch 1.8+
- CUDA 11.0+ (可选，用于GPU加速)

### 安装依赖

```bash
cd MultiSpecVision
pip install -r requirements.txt
```

### 启动Web应用

#### 方式一：直接启动
```bash
# 标准单通道应用
python app.py

# 多通道应用
python multi_channel_app.py
```

#### 方式二：使用启动脚本
```bash
chmod +x run.sh
./run.sh
```

#### 方式三：Docker部署
```bash
# 构建并启动
docker-compose up --build

# 后台运行
docker-compose up -d
```

### 访问应用
- 单通道应用：http://localhost:5000
- 多通道应用：http://localhost:5001

## 模型特性

### Swin Transformer 核心优势
1. **层次化特征表示**：多尺度特征提取
2. **移位窗口机制**：降低计算复杂度
3. **线性计算复杂度**：相对于图像尺寸
4. **灵活的架构**：适应各种视觉任务

### 多通道扩展
1. **动态通道适配**：自动调整输入通道数
2. **通道注意力机制**：智能权重分配
3. **跨通道特征融合**：多模态信息整合
4. **可扩展设计**：支持3-20通道输入

## 技术规格

### 模型参数
- **输入分辨率**：224x224 (可调整)
- **Patch大小**：4x4
- **窗口大小**：7x7
- **层数**：2/2/6/2 (Tiny版本)
- **嵌入维度**：96
- **注意力头数**：3/6/12/24

### 性能指标
- **参数量**：28M (Tiny版本)
- **FLOPs**：4.5G (224x224输入)
- **推理速度**：~50ms (GPU Tesla V100)
- **内存占用**：~2GB (训练时)

## API 使用

### 单张图像推理
```python
from models.multispec_transformer import SwinTransformer
import torch

# 加载模型
model = SwinTransformer()
model.load_state_dict(torch.load('checkpoints/swin_tiny_patch4_window7_224.pth'))

# 推理
with torch.no_grad():
    output = model(input_tensor)
```

### 多通道图像处理
```python
from models.multispec_multichannel import MultiChannelSwinTransformer

# 创建多通道模型
model = MultiChannelSwinTransformer(in_chans=6)  # 6通道输入
output = model(multi_channel_input)
```

## 开发指南

### 微调训练
参考文档：`swin_transformer_二次开发与微调方案.md`

### 多通道开发
参考文档：`swin_transformer_多通道开发需求.md`

### 架构说明
参考文档：`swin_transformer_文件解读说明.md`

## 版权信息

```
Copyright (c) 2024 MultiSpecVision Team
Licensed under the Apache License, Version 2.0
```

## 技术支持

如有技术问题或建议，请联系MultiSpecVision团队。

---

**MultiSpecVision - 让多光谱视觉更智能** 