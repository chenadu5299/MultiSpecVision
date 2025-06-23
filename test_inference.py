#!/usr/bin/env python3
"""
MultiSpecVision model inference test script
"""

import torch
from models.swin_transformer import SwinTransformer

def test_inference():
    """Test model inference functionality"""
    print("Creating MultiSpecVision model...")
    
    # Create model (using tiny version for quick testing)
    model = SwinTransformer()
    
    print("Loading pre-trained weights...")
    try:
        # Load pre-trained weights
        checkpoint = torch.load('checkpoints/swin_tiny_patch4_window7_224.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
    except FileNotFoundError:
        print("⚠️ Pre-trained weights not found, using random weights for testing")
    
    print("Model loading complete!")
    
    # Create random input for testing
    print("\nPerforming inference test...")
    batch_size = 1
    channels = 3  # RGB
    height = width = 224
    
    input_tensor = torch.randn(batch_size, channels, height, width)
    
    # Set to evaluation mode
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
    
    # Get top-5 predictions
    top5_prob, top5_indices = torch.topk(outputs, 5, dim=1)
    
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Output tensor shape: {outputs.shape}")
    print(f"Top-5 predicted classes: {top5_indices[0].tolist()}")
    print(f"Top-5 probabilities: {top5_prob[0].tolist()}")
    
    print("\n✅ MultiSpecVision model inference test successful!")

if __name__ == "__main__":
    test_inference() 