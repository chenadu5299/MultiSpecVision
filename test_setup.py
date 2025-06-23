#!/usr/bin/env python3
"""
MultiSpecVision Environment Test Script
Test whether all necessary components are properly installed and configured
"""

import sys
import os
import torch
import importlib.util

def test_imports():
    """Test necessary module imports"""
    print("=" * 50)
    print("🧪 Testing module imports...")
    
    try:
        from models.multispec_transformer import SwinTransformer
        print("✅ SwinTransformer import successful")
    except ImportError as e:
        print(f"❌ SwinTransformer import failed: {e}")
        return False
    
    try:
        from models.multispec_multichannel import MultiChannelSwinTransformer
        print("✅ MultiChannelSwinTransformer import successful")
    except ImportError as e:
        print(f"❌ MultiChannelSwinTransformer import failed: {e}")
        return False
    
    try:
        import flask
        print("✅ Flask import successful")
    except ImportError as e:
        print(f"❌ Flask import failed: {e}")
        return False
    
    return True

def test_model_creation():
    """Test model creation"""
    print("\n" + "=" * 50)
    print("🔧 Testing model creation...")
    
    try:
        from models.multispec_transformer import SwinTransformer
        model = SwinTransformer()
        print("✅ Standard SwinTransformer model creation successful")
        print(f"   Model parameter count: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"❌ Standard SwinTransformer model creation failed: {e}")
        return False
    
    try:
        from models.multispec_multichannel import MultiChannelSwinTransformer
        model = MultiChannelSwinTransformer(in_chans=6)
        print("✅ Multi-channel SwinTransformer model creation successful (6 channels)")
        print(f"   Model parameter count: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"❌ Multi-channel SwinTransformer model creation failed: {e}")
        return False
    
    return True

def test_checkpoint():
    """Test checkpoint files"""
    print("\n" + "=" * 50)
    print("📂 Testing checkpoint files...")
    
    checkpoint_path = "checkpoints/swin_tiny_patch4_window7_224.pth"
    if os.path.exists(checkpoint_path):
        print(f"✅ Checkpoint file exists: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print(f"   Checkpoint type: {type(checkpoint)}")
            if isinstance(checkpoint, dict):
                print(f"   Checkpoint keys: {list(checkpoint.keys())}")
        except Exception as e:
            print(f"❌ Checkpoint file loading failed: {e}")
            return False
    else:
        print(f"❌ Checkpoint file does not exist: {checkpoint_path}")
        return False
    
    return True

def test_file_structure():
    """Test file structure"""
    print("\n" + "=" * 50)
    print("📁 Testing file structure...")
    
    required_files = [
        "app.py",
        "multi_channel_app.py",
        "test_inference.py",
        "requirements.txt",
        "Dockerfile",
        "docker-compose.yml",
        "models/multispec_transformer.py",
        "models/multispec_multichannel.py",
        "templates/index.html",
        "templates/multi_channel_index.html"
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\nMissing files: {missing_files}")
        return False
    
    return True

def test_system_info():
    """Display system information"""
    print("\n" + "=" * 50)
    print("ℹ️  System information...")
    
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current working directory: {os.getcwd()}")

def main():
    """Main test function"""
    print("🚀 MultiSpecVision Environment Test")
    print("=" * 50)
    
    test_system_info()
    
    tests = [
        ("Module Import", test_imports),
        ("Model Creation", test_model_creation),
        ("Checkpoint Files", test_checkpoint),
        ("File Structure", test_file_structure)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✅ {test_name} test passed")
            else:
                print(f"\n❌ {test_name} test failed")
        except Exception as e:
            print(f"\n❌ {test_name} test error: {e}")
    
    print("\n" + "=" * 50)
    print(f"🎯 Test results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! MultiSpecVision environment is properly configured.")
        print("\n🚀 You can start using the following commands to launch applications:")
        print("   python app.py              # Single-channel application")
        print("   python multi_channel_app.py # Multi-channel application")
        print("   docker-compose up          # Docker deployment")
    else:
        print("⚠️  Some tests failed, please check environment configuration.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 