#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiSpecVision Web应用
提供基于MultiSpecVision模型的图像识别功能
"""

import os
import torch
import torch.nn.functional as F
from PIL import Image
from flask import Flask, request, render_template, jsonify
from models.swin_transformer import SwinTransformer
import torchvision.transforms as transforms

app = Flask(__name__)

# 配置
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# 确保上传目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 加载模型
device = torch.device('cpu')
print("正在加载MultiSpecVision模型...")
model = SwinTransformer(img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                       embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24])

checkpoint = torch.load("checkpoints/swin_tiny_patch4_window7_224.pth", map_location="cpu")
if "model" in checkpoint:
    model.load_state_dict(checkpoint["model"], strict=False)
else:
    model.load_state_dict(checkpoint, strict=False)

model.eval()
print("MultiSpecVision模型加载完成!")

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 简化的类别名称
class_names = [
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead",
    "electric ray", "stingray", "cock", "hen", "ostrich", "brambling",
    "goldfinch", "house finch", "junco", "indigo bunting", "robin",
    "bulbul", "jay", "magpie", "chickadee", "water ouzel", "kite",
    "bald eagle", "vulture", "great grey owl", "European fire salamander",
    "common newt", "eft", "spotted salamander", "axolotl", "bullfrog",
    "tree frog", "tailed frog", "loggerhead", "leatherback turtle",
    "mud turtle", "terrapin", "box turtle", "banded gecko", "common iguana",
    "American chameleon", "whiptail", "agama", "frilled lizard",
    "alligator lizard", "Gila monster", "green lizard", "African chameleon",
    "Komodo dragon", "African crocodile", "American alligator", "triceratops"
] + [f"class_{i}" for i in range(52, 1000)]  # 其余类别用占位符

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': '没有上传文件'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'})
    
    if file and allowed_file(file.filename):
        try:
            # 直接从内存中处理图像
            image = Image.open(file.stream).convert('RGB')
            
            # 预处理
            input_tensor = transform(image).unsqueeze(0)
            
            # 预测
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                
                # 获取top-5预测
                top5_prob, top5_indices = torch.topk(probabilities, 5)
                
                results = []
                for i in range(5):
                    class_idx = top5_indices[0][i].item()
                    prob = top5_prob[0][i].item()
                    class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
                    results.append({
                        'class': class_name,
                        'probability': f"{prob:.4f}",
                        'percentage': f"{prob * 100:.2f}%"
                    })
                
                return jsonify({'results': results})
                
        except Exception as e:
            return jsonify({'error': f'处理图像时出错: {str(e)}'})
    else:
        return jsonify({'error': '不支持的文件格式'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 