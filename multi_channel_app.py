#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Channel MultiSpecVision Web Application
Provides multi-channel image segmentation functionality based on MultiSpecVision architecture
Supports 3-20 channel multi-sensor data processing
"""

import os
import io
import json
import time
import base64
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, jsonify
from models.multi_channel_swin_transformer import MultiChannelSwinTransformer
import torchvision.transforms as transforms

app = Flask(__name__)

# 配置
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'tif', 'npy'}

# 确保上传目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 成像类型到通道数的映射
CHANNEL_MAPPING = {
    'rgb': 3,
    'rgb_ir': 4,
    'multispectral': 8,
    'hyperspectral': 16,
    'thermal': 1,
    'xray': 1
}

# 标准类别名称 (PASCAL VOC 21类)
CLASS_NAMES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
    'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# 生成随机颜色映射
def generate_color_map(num_classes):
    """为每个类别生成随机颜色"""
    np.random.seed(42)  # 固定种子确保颜色一致
    colors = np.random.randint(0, 255, (num_classes, 3))
    colors[0] = [0, 0, 0]  # 背景设为黑色
    return colors

def load_multichannel_image(file_stream, channel_type='rgb', num_channels=3):
    """
    加载多通道图像数据
    支持常规图像文件和numpy数组文件
    """
    try:
        file_stream.seek(0)
        
        # 检查文件类型
        if hasattr(file_stream, 'filename'):
            filename = file_stream.filename.lower()
        else:
            filename = ''
            
        if filename.endswith('.npy'):
            # 处理numpy数组文件
            data = np.load(file_stream)
            if len(data.shape) == 2:  # 单通道
                data = np.expand_dims(data, axis=2)
            elif len(data.shape) == 3 and data.shape[2] > num_channels:
                # 如果通道数超过需要的，选择前N个通道
                data = data[:, :, :num_channels]
            elif len(data.shape) == 3 and data.shape[2] < num_channels:
                # 如果通道数不足，复制最后一个通道
                last_channel = data[:, :, -1:]
                while data.shape[2] < num_channels:
                    data = np.concatenate([data, last_channel], axis=2)
            
            # 确保数据类型为uint8
            if data.dtype != np.uint8:
                data = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
                
            return data
        else:
            # 处理常规图像文件
            image = Image.open(file_stream).convert('RGB')
            data = np.array(image)
            
            if num_channels == 1:
                # 转为灰度图
                data = np.dot(data, [0.299, 0.587, 0.114]).astype(np.uint8)
                data = np.expand_dims(data, axis=2)
            elif num_channels > 3:
                # 如果需要更多通道，复制现有通道
                channels = [data]
                for i in range(num_channels - 3):
                    # 添加处理过的通道（如边缘检测、纹理等）
                    if i % 3 == 0:  # 添加灰度版本
                        gray = np.dot(data, [0.299, 0.587, 0.114]).astype(np.uint8)
                        channels.append(np.expand_dims(gray, axis=2))
                    else:  # 添加其他变换
                        channels.append(data[:, :, i % 3:i % 3 + 1])
                data = np.concatenate(channels, axis=2)
            
            return data
            
    except Exception as e:
        raise ValueError(f"无法加载图像文件: {str(e)}")

# 加载模型
device = torch.device('cpu')
print("正在加载多通道MultiSpecVision模型...")

# 创建模型实例
model = MultiChannelSwinTransformer(
    img_size=224,
    patch_size=4, 
    in_chans=3,  # 默认3通道，会动态调整
    num_classes=21,  # PASCAL VOC 21类
    embed_dim=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=7,
    mlp_ratio=4.,
    qkv_bias=True,
    drop_rate=0.,
    attn_drop_rate=0.,
    drop_path_rate=0.1,
    norm_layer=torch.nn.LayerNorm,
    ape=False,
    patch_norm=True,
    use_checkpoint=False
)

# 尝试加载预训练权重
try:
    model.load_from("checkpoints/swin_tiny_patch4_window7_224.pth", in_chans=3)
    print("成功加载预训练权重")
except Exception as e:
    print(f"警告：无法加载预训练权重 - {e}")

model.eval()
print("多通道MultiSpecVision模型加载完成!")

def create_segmentation_mask(logits, threshold=0.5):
    """创建分割掩码"""
    if len(logits.shape) == 4:  # [B, C, H, W]
        probs = F.softmax(logits, dim=1)
        mask = torch.argmax(probs, dim=1)  # [B, H, W]
        return mask[0].cpu().numpy()  # 返回第一个batch的结果
    else:
        return None

def visualize_segmentation(image, mask, num_classes=21, alpha=0.6):
    """可视化分割结果"""
    if len(image.shape) == 3 and image.shape[2] > 3:
        # 如果是多通道图像，只使用前3个通道进行可视化
        vis_image = image[:, :, :3]
    else:
        vis_image = image
        
    # 确保图像是RGB格式
    if vis_image.shape[2] == 1:
        vis_image = np.repeat(vis_image, 3, axis=2)
    
    color_map = generate_color_map(num_classes)
    
    # 创建彩色分割掩码
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id in range(num_classes):
        colored_mask[mask == class_id] = color_map[class_id]
    
    # 创建叠加图像
    overlay = vis_image.astype(np.float32) * (1 - alpha) + colored_mask.astype(np.float32) * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    return colored_mask, overlay

def preprocess_for_model(image_data, target_size=(224, 224)):
    """预处理图像数据用于模型推理"""
    # 调整图像大小
    if len(image_data.shape) == 3:
        channels = []
        for c in range(image_data.shape[2]):
            channel_img = Image.fromarray(image_data[:, :, c])
            channel_img = channel_img.resize(target_size, Image.LANCZOS)
            channels.append(np.array(channel_img))
        
        resized_data = np.stack(channels, axis=2)
    else:
        img = Image.fromarray(image_data)
        resized_data = np.array(img.resize(target_size, Image.LANCZOS))
    
    # 转换为PyTorch张量
    if len(resized_data.shape) == 2:
        resized_data = np.expand_dims(resized_data, axis=2)
    
    # 归一化到[0,1]
    tensor_data = torch.from_numpy(resized_data).float() / 255.0
    
    # 调整维度顺序为[C, H, W]
    if len(tensor_data.shape) == 3:
        tensor_data = tensor_data.permute(2, 0, 1)
    
    # 添加batch维度
    tensor_data = tensor_data.unsqueeze(0)
    
    return tensor_data

def image_to_base64(image_array):
    """将numpy数组转换为base64字符串"""
    if len(image_array.shape) == 2:
        image_array = np.expand_dims(image_array, axis=2)
    
    img = Image.fromarray(image_array.astype(np.uint8))
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

@app.route('/')
def index():
    return render_template('multi_channel_index.html')

@app.route('/segment', methods=['POST'])
def segment():
    if 'file' not in request.files:
        return jsonify({'error': '没有上传文件'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'})
    
    if not allowed_file(file.filename):
        return jsonify({'error': '不支持的文件格式'})
    
    try:
        start_time = time.time()
        
        # 获取参数
        channel_type = request.form.get('channel_type', 'rgb')
        num_channels = int(request.form.get('num_channels', '3'))
        task_type = request.form.get('task_type', 'semantic')
        num_classes = int(request.form.get('num_classes', '21'))
        confidence_threshold = float(request.form.get('confidence_threshold', '0.5'))
        
        # 如果是预设类型，使用映射表
        if channel_type != 'custom':
            num_channels = CHANNEL_MAPPING.get(channel_type, 3)
        
        # 加载多通道图像
        image_data = load_multichannel_image(file.stream, channel_type, num_channels)
        
        # 动态调整模型输入通道
        global model
        if model.patch_embed.proj.in_channels != num_channels:
            print(f"调整模型输入通道数从 {model.patch_embed.proj.in_channels} 到 {num_channels}")
            model.adjust_input_channels(num_channels)
        
        # 预处理图像
        input_tensor = preprocess_for_model(image_data)
        
        # 模型推理
        with torch.no_grad():
            outputs = model(input_tensor)
            
            # 创建分割掩码
            segmentation_mask = create_segmentation_mask(outputs, confidence_threshold)
            
            if segmentation_mask is None:
                return jsonify({'error': '模型输出格式错误'})
        
        # 计算处理时间
        processing_time = round(time.time() - start_time, 2)
        
        # 统计检测到的对象数量
        unique_classes = np.unique(segmentation_mask)
        detected_objects = len(unique_classes) - 1 if 0 in unique_classes else len(unique_classes)
        
        # 生成可视化结果
        colored_segmentation, overlay_result = visualize_segmentation(
            image_data, segmentation_mask, num_classes, alpha=0.6
        )
        
        # 准备响应数据
        response_data = {
            'success': True,
            'processing_time': processing_time,
            'detected_objects': detected_objects,
            'num_channels_used': num_channels,
            'channel_type': channel_type,
            'task_type': task_type,
            'confidence_threshold': confidence_threshold,
            
            # 图像数据（base64编码）
            'original_image': image_to_base64(image_data[:, :, :3] if image_data.shape[2] >= 3 else image_data),
            'segmentation_mask': image_to_base64(segmentation_mask),
            'colored_segmentation': image_to_base64(colored_segmentation),
            'overlay_result': image_to_base64(overlay_result),
            
            # 统计信息
            'class_distribution': {
                CLASS_NAMES[i] if i < len(CLASS_NAMES) else f'class_{i}': 
                int(np.sum(segmentation_mask == i))
                for i in unique_classes
            },
            'image_shape': image_data.shape,
            'output_shape': segmentation_mask.shape
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': f'处理图像时出错: {str(e)}'})

@app.route('/health')
def health():
    """健康检查端点"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'supported_channels': list(CHANNEL_MAPPING.keys()) + ['custom'],
        'max_channels': 20
    })

# 错误处理
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': '文件太大，请上传小于16MB的文件'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': '服务器内部错误'}), 500

if __name__ == '__main__':
    # 全局加载模型 - 使用原始的多通道MultiSpecVision模型
    try:
        print("尝试加载多通道MultiSpecVision模型...")
        model = model  # 使用已经加载的模型
        print("成功加载多通道MultiSpecVision模型")
    except Exception as e:
        print(f"模型加载失败: {e}")
        exit(1)
    
    # 启动Flask应用
    print("启动多通道MultiSpecVision Web应用...")
    app.run(host='0.0.0.0', port=5000, debug=False) 