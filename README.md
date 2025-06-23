# MultiSpecVision Web应用

这是一个简单的Web应用，用于演示MultiSpecVision模型的图像识别能力。

## 功能

- 上传图片并使用MultiSpecVision模型进行识别
- 显示Top-5预测结果及其概率
- 简洁直观的用户界面

## 运行方式

### 方法1：使用Docker（推荐）

1. 确保已安装Docker和Docker Compose
2. 在项目根目录下运行：

```bash
docker-compose up --build
```

3. 打开浏览器访问：http://localhost:5000

### 方法2：直接运行（需要Python环境）

1. 确保已安装Python 3.7+和pip
2. 安装依赖：

```bash
pip install torch torchvision flask pillow
```

3. 确保已下载预训练模型：

```bash
mkdir -p checkpoints
wget -P checkpoints https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
```

4. 运行应用：

```bash
python app.py
```

5. 打开浏览器访问：http://localhost:5000

## 项目结构

- `app.py`：Flask应用主文件
- `templates/index.html`：Web界面模板
- `models/`：MultiSpecVision模型代码
- `checkpoints/`：预训练模型权重
- `Dockerfile`和`docker-compose.yml`：Docker配置文件

## 注意事项

- 当前仅包含部分ImageNet类别名称，完整版本需要加载imagenet_classes.txt
- 在M1/M2芯片的Mac上运行时，可能会看到平台不匹配警告，但不影响功能
- 首次运行时需要下载预训练模型（约110MB） 