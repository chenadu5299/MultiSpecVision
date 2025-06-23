# Multi-Channel MultiSpecVision Image Segmentation System

This project is based on the MultiSpecVision architecture and implements an image segmentation system that supports multi-channel input (3-20 channels), capable of processing various imaging types including RGB, multispectral, thermal imaging, X-ray, and depth imaging.

## Core Features

- **Multi-Channel Support**: Dynamic support for 3-20 channel input data
- **Multi-Sensor Fusion**: Supports RGB, multispectral, hyperspectral, thermal imaging, X-ray and other imaging modes
- **Channel Attention Mechanism**: Adaptive learning of importance weights for different channels
- **Flexible Segmentation Tasks**: Supports semantic segmentation, instance segmentation, and panoptic segmentation
- **Pre-trained Weight Utilization**: Can load weights from standard MultiSpecVision pre-trained models
- **Real-time Web Interface**: Provides intuitive multi-channel image segmentation interface

## Technical Architecture

1. **Multi-Channel MultiSpecVision Model**: Core model supporting dynamic channel input and image segmentation output
2. **Channel Adaptive Layer**: Dynamically adjusts model structure based on input channel count
3. **Multi-Modal Preprocessing**: Specialized preprocessing pipelines for different imaging types
4. **Segmentation Post-processing**: Includes segmentation mask generation, visualization, and result export

## Project Structure

```
├── models/
│   ├── multi_channel_swin_transformer.py  # Multi-channel MultiSpecVision core model
│   ├── swin_transformer.py                # Base MultiSpecVision model
│   └── dng_processor.py                   # DNG raw image processor
├── templates/
│   ├── index.html                         # Basic image recognition interface
│   └── multi_channel_index.html           # Multi-channel segmentation interface
├── multi_channel_app.py                   # Multi-channel Web application main program
├── app.py                                 # Basic Web application
├── test_inference.py                      # Model inference testing
├── checkpoints/                           # Pre-trained model weights
├── Dockerfile                            # Docker container configuration
├── docker-compose.yml                    # Docker orchestration configuration
└── requirements.txt                       # Python dependencies
```

## Quick Start

### Method 1: Docker Deployment (Recommended)

1. Ensure Docker and Docker Compose are installed

2. Clone the project and enter directory:
```bash
git clone <repository_url>
cd multispecvision
```

3. Build and start containers:
```bash
docker-compose up --build
```

4. Access Web interface:
- Basic image recognition: http://localhost:5000
- Multi-channel segmentation: http://localhost:5000 (select multi-channel mode)

### Method 2: Local Environment

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Download pre-trained model:
```bash
mkdir -p checkpoints
wget -P checkpoints https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
```

3. Start application:
```bash
# Basic application
python app.py

# Multi-channel segmentation application
python multi_channel_app.py
```

## Usage Guide

### Supported Imaging Types

| Imaging Type | Channels | Application Scenarios | Examples |
|-------------|----------|---------------------|----------|
| RGB | 3 | Standard color images | Natural images, surveillance video |
| RGB+Infrared | 4 | Multispectral imaging | Agricultural monitoring, building detection |
| Multispectral | 8 | Remote sensing imaging | Satellite images, geological exploration |
| Hyperspectral | 16 | Precision detection | Food safety, material analysis |
| Thermal imaging | 1 | Temperature detection | Medical diagnosis, industrial inspection |
| X-ray imaging | 1 | Medical imaging | Fracture detection, lung screening |
| Custom | 1-20 | Special applications | Research, custom sensors |

### Multi-Channel Data Format

Supports the following file formats:
- **Standard Images**: JPG, PNG, BMP, TIFF (automatically expanded to specified channel count)
- **Multi-Channel Data**: NPY format numpy arrays (shape H×W×C)
- **Raw Data**: Multi-page TIFF files, DNG format

### Segmentation Task Types

1. **Semantic Segmentation**: Pixel-level classification, same class pixels have same label
2. **Instance Segmentation**: Distinguish different individuals of the same class
3. **Panoptic Segmentation**: Combines semantic and instance segmentation

### API Usage Example

```python
from models.multispec_multichannel import MultiChannelSwinTransformer
import torch
import numpy as np

# Create model
model = MultiChannelSwinTransformer(
    img_size=224,
    in_chans=8,  # 8-channel input
    num_classes=21,
    embed_dim=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24]
)

# Load pre-trained weights
model.load_from("checkpoints/swin_tiny_patch4_window7_224.pth", in_chans=3)

# Inference
input_data = torch.randn(1, 8, 224, 224)  # batch×channels×height×width
with torch.no_grad():
    outputs = model(input_data)
    segmentation = torch.argmax(outputs, dim=1)
```

## Technical Features

### 1. Dynamic Channel Adaptation

The model can dynamically adjust based on input data channel count:
- Channel expansion: Automatically replicate channels when fewer than expected
- Channel selection: Intelligently select key channels when more than expected
- Weight initialization: New channels use weighted average of pre-trained weights

### 2. Channel Attention Mechanism

```python
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )
```

### 3. Multi-Scale Feature Fusion

- **Pyramid Features**: Extract feature representations at different resolutions
- **Cross-Scale Connections**: Maintain detail information while obtaining global semantics
- **Adaptive Fusion**: Adjust fusion strategy based on task type

## Performance Optimization

### Memory Optimization
- **Gradient Checkpointing**: Reduce memory usage during training
- **Mixed Precision**: Support FP16 inference acceleration
- **Batch Adaptation**: Dynamically adjust batch size based on GPU memory

### Speed Optimization
- **Model Quantization**: Support INT8 quantized deployment
- **Operator Fusion**: Reduce forward propagation computational overhead
- **Caching Mechanism**: Reuse computation results to avoid redundant calculations

## Deployment Configuration

### Docker Configuration Optimization

```dockerfile
# Configuration optimized for multi-channel processing
ENV OPENCV_THREAD_COUNT=4
ENV MKL_NUM_THREADS=4
ENV NUMEXPR_MAX_THREADS=4
```

### Production Environment Configuration

```yaml
# docker-compose.prod.yml
services:
  multispecvision:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'
```

## Important Notes

1. **Memory Requirements**: Multi-channel processing requires more memory, recommend at least 4GB available memory
2. **Pre-trained Weights**: First run will automatically download approximately 110MB pre-trained model
3. **Platform Compatibility**: May see platform warnings on M1/M2 Mac, but functionality is not affected
4. **File Size Limit**: Web interface defaults to 16MB file size limit, adjustable in configuration

## License

This project is open source under the MIT License, see LICENSE file for details.

## Contribution Guidelines

Welcome to submit Issues and Pull Requests! Before contributing code, please ensure:
1. Code follows project style guidelines
2. Add necessary test cases
3. Update relevant documentation 