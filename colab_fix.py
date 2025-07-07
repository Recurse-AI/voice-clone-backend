#!/usr/bin/env python3
"""
Google Colab Compatibility Fix for Voice Cloning API
Fixes protobuf and tensorflow version conflicts
"""

import subprocess
import sys
import os

def run_command(cmd):
    """Run command and print output"""
    print(f"🔧 Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        return False
    print(f"✅ Success: {result.stdout}")
    return True

def fix_protobuf_compatibility():
    """Fix protobuf compatibility issues"""
    print("🔧 Fixing protobuf compatibility issues...")
    
    # Uninstall conflicting packages
    commands = [
        "pip uninstall -y protobuf tensorflow tensorflow-cpu",
        "pip install protobuf==3.20.3",
        "pip install tensorflow==2.12.0",
        "pip install --upgrade transformers",
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    ]
    
    for cmd in commands:
        if not run_command(cmd):
            print(f"❌ Failed to run: {cmd}")
            return False
    
    return True

def test_imports():
    """Test if imports work after fixes"""
    print("🧪 Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
        
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Google Colab Compatibility Fix Starting...")
    
    # Fix compatibility issues
    if not fix_protobuf_compatibility():
        print("❌ Failed to fix compatibility issues")
        return False
    
    # Test imports
    if not test_imports():
        print("❌ Import tests failed")
        return False
    
    print("✅ All compatibility issues fixed!")
    print("🔄 Please restart runtime now:")
    print("   Runtime > Restart Runtime")
    
    return True

if __name__ == "__main__":
    main() 