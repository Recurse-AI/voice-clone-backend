#!/bin/bash

echo "🚀 Setting up Voice Cloning API Environment..."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FFMPEG_USE_GPU=1
export CUDA_LAUNCH_BLOCKING=0
export DEBIAN_FRONTEND=noninteractive

echo "🧹 Cleaning temporary directories..."
rm -rf /tmp/* /logs/* 2>/dev/null || true
chmod 1777 /tmp 2>/dev/null || true
export TMPDIR=/tmp

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
    lsof \
    2>/dev/null || echo "Some packages might already be installed"

echo "🚀 Checking CUDA/CUDNN libraries..."
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected, checking CUDA libraries..."
    if ldconfig -p | grep -q cudnn; then
        echo "✅ CUDNN libraries already installed"
    else
        echo "⚠️ CUDNN libraries not found, installing..."
        apt-get update -y || true
        apt-get install -y libcudnn8 libcudnn8-dev 2>/dev/null || true
    fi

    export CUDA_HOME=/usr/local/cuda
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    export PATH=/usr/local/cuda/bin:$PATH
    echo "✅ CUDA/CUDNN check completed"
else
    echo "⚠️ No GPU detected, skipping CUDA installation"
fi

echo "🔍 Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️ No GPU detected, running on CPU"
fi

echo "📂 Creating directories..."
mkdir -p ./tmp ./logs
chmod 755 ./tmp ./logs

echo "🐍 Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip
apt-get install fonts-noto fonts-noto-extra

echo "📋 Installing Python dependencies..."
pip install -r requirements.txt

echo "🎭 Installing Playwright browsers..."
python -m playwright install --with-deps chromium

echo "🔐 Installing YouTube PO Token provider..."
pip install --upgrade bgutil-ytdlp-pot-provider --quiet || true

echo "🐟 Setting up Fish Speech..."
if [ ! -d "fish-speech" ]; then
    git clone https://github.com/fishaudio/fish-speech.git
fi

cd fish-speech
pip install -e . --no-deps
cd ..

export PYTHONPATH="${PWD}/fish-speech:${PYTHONPATH}"

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
        for i in range(torch.cuda.device_count()):
            print(f'  Device {i}: {torch.cuda.get_device_name(i)}')
except Exception as e:
    print(f'PyTorch check failed: {e}')

try:
    import fastapi
    print(f'FastAPI: {fastapi.__version__}')
except:
    pass

try:
    import redis
    print(f'Redis Python client: {redis.__version__}')
except:
    pass
"

if [ ! -z "${HF_TOKEN}" ]; then
    echo "🤗 Setting up Hugging Face models..."
    pip install huggingface_hub
    echo "${HF_TOKEN}" | huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential || true
    
    mkdir -p checkpoints
    echo "📥 Downloading model checkpoints..."
    huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini || true
fi

echo ""
echo "🎉 Environment setup completed!"
echo ""
echo "🔧 Next steps:"
echo "  1. Start Redis: sudo service redis-server start"
echo "  2. Run main setup: ./runpod_setup.sh"
