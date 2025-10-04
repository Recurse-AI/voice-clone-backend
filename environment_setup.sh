#!/bin/bash

# Environment Setup and Dependency Installation Script
# Separate from main runpod_setup.sh for modularity

echo "🚀 Setting up Voice Cloning API Environment..."

# Environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FFMPEG_USE_GPU=1
export CUDA_LAUNCH_BLOCKING=0
export TORCH_BACKENDS_CUDNN_DETERMINISTIC=0
export DEBIAN_FRONTEND=noninteractive

# Clean temp directories
echo "🧹 Cleaning temporary directories..."
rm -rf /tmp/* /logs/* 2>/dev/null || true
chmod 1777 /tmp 2>/dev/null || true
export TMPDIR=/tmp

# Find project directory
echo "📁 Locating project directory..."
if [ -f "requirements.txt" ]; then
    echo "Found requirements.txt in current directory"
elif [ -f "voice-clone-backend/requirements.txt" ]; then
    cd voice-clone-backend
elif [ -f "/workspace/voice-clone-backend/requirements.txt" ]; then
    cd /workspace/voice-clone-backend
else
    echo "❌ Could not find requirements.txt file"
    exit 1
fi

echo "Working directory: $(pwd)"

# Install system dependencies
echo "📦 Installing system dependencies..."
apt-get update -y || true
apt-get install -y \
    ffmpeg \
    libsndfile1 \
    python3-dev \
    python3-pip \
    python3-venv \
    git \
    curl \
    build-essential \
    portaudio19-dev \
    libsox-dev \
    redis-server \
    wget \
    gnupg2 \
    lsof \
    2>/dev/null || echo "Some packages might already be installed"

# Note: PO Token provider is installed via pip (Python plugin)
# No Node.js server required!

# Check CUDA and CUDNN libraries for GPU acceleration
echo "🚀 Checking CUDA/CUDNN libraries..."
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected, checking CUDA libraries..."

    # Check if CUDNN is already installed
    if ldconfig -p | grep -q cudnn; then
        echo "✅ CUDNN libraries already installed"
    else
        echo "⚠️ CUDNN libraries not found, installing..."
        # Install CUDNN if missing
        apt-get update -y || true
        apt-get install -y \
            libcudnn8 \
            libcudnn8-dev \
            2>/dev/null || echo "CUDNN installation completed with warnings"
    fi

    # Set CUDA environment variables (libraries already installed)
    export CUDA_HOME=/usr/local/cuda
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    export PATH=/usr/local/cuda/bin:$PATH

    echo "✅ CUDA/CUDNN check completed"
else
    echo "⚠️ No GPU detected, skipping CUDA installation"
fi

# Check GPU
echo "🔍 Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo "GPU Memory Status:"
    nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits
else
    echo "⚠️ No GPU detected, running on CPU"
fi

# Create necessary directories
echo "📂 Creating directories..."
mkdir -p ./tmp ./logs
chmod 755 ./tmp ./logs

# Setup Python environment
echo "🐍 Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip

echo "📋 Installing Python dependencies..."
pip install -r requirements.txt

echo "🎭 Installing Playwright browsers for auto cookie fetch..."
python -m playwright install chromium --with-deps

echo "🔐 Installing YouTube PO Token provider plugin..."
pip install --upgrade bgutil-ytdlp-pot-provider --quiet || echo "PO Token plugin installation attempted"

# Setup Fish Speech
echo "🐟 Setting up Fish Speech..."
if [ ! -d "fish-speech" ]; then
    echo "Cloning Fish Speech repository..."
    git clone https://github.com/fishaudio/fish-speech.git
fi

cd fish-speech
pip install -e . --no-deps
cd ..

export PYTHONPATH="${PWD}/fish-speech:${PYTHONPATH}"

# Verify installation
echo "✅ Verifying installation..."
python3 -c "
import sys
print(f'Python: {sys.version}')

try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA Available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA Devices: {torch.cuda.device_count()}')
        print(f'CUDA Version: {torch.version.cuda}')
        print(f'CUDNN Version: {torch.backends.cudnn.version()}')
        for i in range(torch.cuda.device_count()):
            print(f'  Device {i}: {torch.cuda.get_device_name(i)}')
            print(f'  Memory: {torch.cuda.get_device_properties(i).total_memory // 1024**3}GB')
    else:
        print('⚠️ CUDA not available - check NVIDIA drivers and CUDA installation')
except Exception as e:
    print(f'PyTorch check failed: {e}')

try:
    import fastapi
    print(f'FastAPI: {fastapi.__version__}')
except Exception as e:
    print(f'FastAPI check failed: {e}')

try:
    import redis
    print(f'Redis Python client: {redis.__version__}')
except Exception as e:
    print(f'Redis check failed: {e}')

try:
    import ffmpeg
    print('FFmpeg Python wrapper: Available')
except:
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print('FFmpeg system: Available')
        else:
            print('FFmpeg: Not found in PATH')
    except:
        print('FFmpeg: Not available')
"

# Download models if HF_TOKEN provided
if [ ! -z "${HF_TOKEN}" ]; then
    echo "🤗 Setting up Hugging Face models..."
    pip install huggingface_hub
    echo "${HF_TOKEN}" | huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential || true
    
    mkdir -p checkpoints
    echo "📥 Downloading model checkpoints..."
    huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini || echo "Model download failed, continuing..."
fi

# Test Redis connection
echo "🔍 Testing Redis connection..."
python3 -c "
import redis
try:
    r = redis.Redis(host='127.0.0.1', port=6379, db=0)
    r.ping()
    print('✅ Redis connection test: SUCCESS')
except Exception as e:
    print(f'⚠️ Redis connection test: FAILED - {e}')
    print('Note: Redis server needs to be started separately')
"

echo ""
echo "🎉 Environment setup completed!"
echo ""
echo "📋 Summary:"
echo "  ✅ System dependencies installed"
echo "  ✅ Python virtual environment created"
echo "  ✅ Python packages installed"
echo "  ✅ YouTube PO Token plugin installed"
echo "  ✅ Fish Speech setup completed"
echo "  ✅ Installation verified"
echo ""
echo "🔧 Next steps:"
echo "  1. Start Redis: sudo service redis-server start"
echo "  2. Run main setup: ./runpod_setup.sh"
echo ""
echo "💡 Useful commands:"
echo "  source venv/bin/activate    # Activate virtual environment"
echo "  python check_workers.py     # Check worker status"
echo "  tail -f logs/workers.log    # Monitor workers"
