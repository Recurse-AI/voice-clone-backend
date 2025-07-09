#!/bin/bash

# Install system dependencies
apt-get update
apt-get install -y ffmpeg libsndfile1 git

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p /tmp/voice_cloning
mkdir -p /workspace/outputs

# Set environment variables
export CUDA_AVAILABLE=true
export PYTHONPATH="/workspace/voice-clone-backend:$PYTHONPATH"

# Create .env file if not exists
if [ ! -f .env ]; then
    echo "⚙️ Creating .env file..."
    cat > .env << EOF
# API Configuration
API_TITLE=Voice Cloning API
API_VERSION=1.0.0
HOST=0.0.0.0
PORT=8000

# Model Configuration
DIA_DEVICE=cuda
CUDA_AVAILABLE=true

# Storage Configuration (Optional)
R2_ACCESS_KEY_ID=${R2_ACCESS_KEY_ID:-}
R2_SECRET_ACCESS_KEY=${R2_SECRET_ACCESS_KEY:-}
R2_BUCKET_NAME=${R2_BUCKET_NAME:-}
R2_ENDPOINT_URL=${R2_ENDPOINT_URL:-}

# API Keys (Optional)
OPENAI_API_KEY=${OPENAI_API_KEY:-}
ASSEMBLYAI_API_KEY=${ASSEMBLYAI_API_KEY:-}

# RunPod Configuration
API_ACCESS_TOKEN=${API_ACCESS_TOKEN:-}
RUNPOD_ENDPOINT_ID=${RUNPOD_ENDPOINT_ID:-}
RUNPOD_TIMEOUT=${RUNPOD_TIMEOUT:-}
EOF
fi

# Check GPU availability
if nvidia-smi > /dev/null 2>&1; then
    echo "🖥️ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
else
    echo "⚠️ No GPU detected! This application requires GPU."
    exit 1
fi

# Start the API
echo "🎵 Starting Voice Cloning API..."
python main.py 