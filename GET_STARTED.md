# Getting Started with MultiSpecVision

Welcome to MultiSpecVision! This guide will help you get up and running quickly with our multi-channel image processing and segmentation system.

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### System Requirements
- **Memory**: 4GB RAM minimum, 8GB recommended
- **GPU**: Optional but recommended (CUDA-compatible)
- **Storage**: 2GB free space

## 🚀 Quick Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd MultiSpecVision
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python test_setup.py
```

## 🎯 Quick Start Examples

### Single-Channel Image Recognition

1. **Start the application**
   ```bash
   python app.py
   ```

2. **Open your browser**
   - Navigate to `http://localhost:5000`
   - Upload an image (JPG, PNG, GIF, BMP)
   - Click "Start Recognition"

### Multi-Channel Image Segmentation

1. **Start the multi-channel application**
   ```bash
   python multi_channel_app.py
   ```

2. **Configure your setup**
   - Choose imaging type (RGB, Multispectral, etc.)
   - Set channel count (3-20 channels)
   - Select segmentation task type

3. **Process your data**
   - Upload multi-channel image
   - Adjust confidence threshold
   - Run segmentation

## 🔧 Configuration Options

### Environment Variables
```bash
export MULTISPEC_MODEL_PATH="./checkpoints/"
export MULTISPEC_LOG_LEVEL="INFO"
export MULTISPEC_DEVICE="cuda"  # or "cpu"
```

### Model Configuration
- **Input Channels**: 3-20 channels supported
- **Image Size**: 224x224 default (configurable)
- **Batch Size**: Adjustable based on available memory

## 📊 Usage Examples

### Python API Usage

```python
from models.multispec_transformer import SwinTransformer
from models.multispec_multichannel import MultiChannelSwinTransformer
import torch

# Single-channel model
model = SwinTransformer()
model.eval()

# Multi-channel model (example: 6 channels)
multi_model = MultiChannelSwinTransformer(in_chans=6)
multi_model.eval()

# Example inference
with torch.no_grad():
    # Single-channel: (batch, 3, height, width)
    single_input = torch.randn(1, 3, 224, 224)
    single_output = model(single_input)
    
    # Multi-channel: (batch, 6, height, width)
    multi_input = torch.randn(1, 6, 224, 224)
    multi_output = multi_model(multi_input)
```

### Command Line Usage

```bash
# Test model inference
python test_inference.py

# Run environment diagnostics
python test_setup.py

# Start web applications
python app.py              # Single-channel
python multi_channel_app.py # Multi-channel
```

## 🐳 Docker Deployment

### Quick Docker Setup
```bash
# Build the image
docker build -t multispecvision .

# Run the container
docker run -p 5000:5000 multispecvision
```

### Docker Compose
```bash
docker-compose up
```

## 📁 Project Structure

```
MultiSpecVision/
├── models/                          # Core model implementations
│   ├── multispec_transformer.py     # Base transformer model
│   ├── multispec_multichannel.py    # Multi-channel variant
│   └── dng_processor.py            # DNG file processor
├── templates/                       # Web interface templates
│   ├── index.html                  # Single-channel UI
│   └── multi_channel_index.html    # Multi-channel UI
├── checkpoints/                     # Pre-trained models
├── app.py                          # Single-channel web app
├── multi_channel_app.py            # Multi-channel web app
├── test_setup.py                   # Environment tests
├── test_inference.py              # Model tests
└── requirements.txt                # Dependencies
```

## 🎛️ Advanced Configuration

### Custom Model Training
```python
# Example training setup
from models.multispec_transformer import SwinTransformer

model = SwinTransformer(
    img_size=224,
    patch_size=4,
    in_chans=3,
    num_classes=1000,
    embed_dim=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24]
)
```

### Multi-Channel Processing
```python
from models.multispec_multichannel import MultiChannelSwinTransformer

# Configure for your specific imaging setup
model = MultiChannelSwinTransformer(
    in_chans=8,              # Number of input channels
    num_classes=21,          # Segmentation classes
    channel_attention=True,   # Enable channel attention
    fusion_strategy='early'   # or 'late', 'middle'
)
```

## 🚨 Troubleshooting

### Common Issues

1. **ImportError: No module named 'torch'**
   ```bash
   pip install torch torchvision
   ```

2. **CUDA out of memory**
   - Reduce batch size
   - Use CPU mode: `export MULTISPEC_DEVICE="cpu"`

3. **Model file not found**
   - Ensure checkpoints are in the correct directory
   - Download required model weights

4. **Port already in use**
   ```bash
   # Use different port
   python app.py --port 5001
   ```

### Performance Optimization

- **GPU Memory**: Monitor with `nvidia-smi`
- **CPU Usage**: Use multiple workers for data loading
- **Inference Speed**: Enable mixed precision training

## 📚 Next Steps

1. **Explore Documentation**
   - [Architecture Guide](Architecture_Documentation.md)
   - [Development Guide](Development_Fine_Tuning_Guide.md)
   - [Multi-Channel Requirements](Multi_Channel_Development_Requirements.md)

2. **Try Advanced Features**
   - Custom channel configurations
   - Batch processing
   - Model fine-tuning

3. **Join the Community**
   - Report issues and suggest features
   - Contribute code improvements
   - Share your use cases

## 📞 Getting Help

- **Documentation**: Check the `docs/` directory
- **Issues**: Open a GitHub issue
- **Discussions**: Join community discussions
- **Support**: See [SUPPORT.md](SUPPORT.md)

Happy processing with MultiSpecVision! 🎉
