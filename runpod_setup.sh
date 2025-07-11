#!/bin/bash

# Platform independent setup script for Voice Cloning API
# Works on Linux (RunPod) and Windows (Git Bash/WSL)

echo "🚀 Voice Cloning API Setup - Platform Independent"

# Detect platform
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    PLATFORM="windows"
    echo "📱 Detected: Windows"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLATFORM="linux"
    echo "🐧 Detected: Linux"
else
    PLATFORM="unknown"
    echo "❓ Unknown platform: $OSTYPE"
fi

# Clean previous setup
echo "🧹 Cleaning previous setup..."
rm -rf /tmp/voice_cloning/* 2>/dev/null || true
rm -rf ./tmp/voice_cloning/* 2>/dev/null || true
rm -rf /workspace/outputs/* 2>/dev/null || true
rm -f .env
rm -rf __pycache__
rm -rf */__pycache__
rm -rf .pytest_cache
rm -rf *.pyc
rm -rf */*.pyc

# Install system dependencies based on platform
echo "📦 Installing system dependencies..."

if [[ "$PLATFORM" == "linux" ]]; then
    # Linux (RunPod) setup
    echo "🐧 Installing Linux dependencies..."
    apt-get update
    apt-get install -y ffmpeg libsndfile1 git wget curl
    
    # Create directories
    mkdir -p /tmp/voice_cloning
    mkdir -p /workspace/outputs
    
    # Set environment variables
    export CUDA_AVAILABLE=true
    export PYTHONPATH="/workspace/voice-clone-backend:$PYTHONPATH"
    
elif [[ "$PLATFORM" == "windows" ]]; then
    # Windows setup
    echo "🪟 Installing Windows dependencies..."
    
    # Check if FFmpeg is installed
    if ! command -v ffmpeg &> /dev/null; then
        echo "📥 FFmpeg not found. Installing FFmpeg..."
        
        # Download FFmpeg for Windows
        FFMPEG_VERSION="6.1"
        FFMPEG_URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
        
        echo "📥 Downloading FFmpeg..."
        curl -L -o ffmpeg.zip "$FFMPEG_URL"
        
        echo "📦 Extracting FFmpeg..."
        unzip -q ffmpeg.zip
        mv ffmpeg-master-latest-win64-gpl ffmpeg
        
        # Add to PATH for current session
        export PATH="$PWD/ffmpeg/bin:$PATH"
        
        echo "✅ FFmpeg installed to: $PWD/ffmpeg/bin"
    else
        echo "✅ FFmpeg already installed"
    fi
    
    # Create directories for Windows
    mkdir -p ./tmp/voice_cloning
    mkdir -p ./outputs
    
    # Set environment variables for Windows
    export CUDA_AVAILABLE=true
    export TEMP_DIR="./tmp/voice_cloning"
fi

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt

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

# Storage Configuration (Required for video processing)
R2_ACCESS_KEY_ID=${R2_ACCESS_KEY_ID:-}
R2_SECRET_ACCESS_KEY=${R2_SECRET_ACCESS_KEY:-}
R2_BUCKET_NAME=${R2_BUCKET_NAME:-}
R2_ENDPOINT_URL=${R2_ENDPOINT_URL:-}
R2_REGION=auto
R2_PUBLIC_URL=${R2_PUBLIC_URL:-}

# API Keys (Required for processing)
OPENAI_API_KEY=${OPENAI_API_KEY:-}
ASSEMBLYAI_API_KEY=${ASSEMBLYAI_API_KEY:-}

# RunPod Configuration (Required for audio separation)
API_ACCESS_TOKEN=${API_ACCESS_TOKEN:-}
RUNPOD_ENDPOINT_ID=${RUNPOD_ENDPOINT_ID:-}
RUNPOD_TIMEOUT=${RUNPOD_TIMEOUT:-1800000}

# Processing Options
ENABLE_SUBTITLES=true
ENABLE_INSTRUMENTS=true

# Platform specific settings
TEMP_DIR=${TEMP_DIR:-/tmp/voice_cloning}
EOF
fi

# Check if required environment variables are set
echo "🔍 Checking environment variables..."
if [ -z "$R2_ACCESS_KEY_ID" ] || [ -z "$R2_SECRET_ACCESS_KEY" ] || [ -z "$R2_BUCKET_NAME" ]; then
    echo "⚠️ Warning: R2 Storage credentials not set. Video processing may fail."
    echo "   Please set: R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME"
fi

if [ -z "$ASSEMBLYAI_API_KEY" ]; then
    echo "⚠️ Warning: AssemblyAI API key not set. Voice cloning may fail."
    echo "   Please set: ASSEMBLYAI_API_KEY"
fi

if [ -z "$API_ACCESS_TOKEN" ] || [ -z "$RUNPOD_ENDPOINT_ID" ]; then
    echo "⚠️ Warning: RunPod credentials not set. Audio separation may fail."
    echo "   Please set: API_ACCESS_TOKEN, RUNPOD_ENDPOINT_ID"
fi

# Check GPU availability
echo "🖥️ Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    export CUDA_AVAILABLE=true
else
    echo "⚠️ No GPU detected! This application requires GPU for optimal performance."
    echo "   Continuing with CPU (may be slow)..."
    export CUDA_AVAILABLE=false
fi

# Test FFmpeg installation
echo "🎬 Testing FFmpeg installation..."
if command -v ffmpeg &> /dev/null; then
    ffmpeg -version | head -1
    echo "✅ FFmpeg is working"
else
    echo "❌ FFmpeg not found! Please install FFmpeg manually."
    if [[ "$PLATFORM" == "windows" ]]; then
        echo "   Windows: Download from https://ffmpeg.org/download.html"
        echo "   Or use: choco install ffmpeg"
    fi
    exit 1
fi

# Kill any existing processes
echo "🔄 Stopping any existing processes..."
if [[ "$PLATFORM" == "linux" ]]; then
    pkill -f "python main.py" || true
    pkill -f "uvicorn" || true
else
    taskkill /F /IM python.exe 2>/dev/null || true
fi
sleep 2

# Start the API in background
echo "🎵 Starting Voice Cloning API in background..."
if [[ "$PLATFORM" == "linux" ]]; then
    nohup python main.py > api.log 2>&1 &
else
    python main.py > api.log 2>&1 &
fi

echo "✅ API started in background. Logs: api.log"

# Show status
echo "📊 API Status:"
if [[ "$PLATFORM" == "linux" ]]; then
    ps aux | grep "python main.py" | grep -v grep || echo "❌ API not running"
else
    tasklist | findstr python || echo "❌ API not running"
fi

echo "📋 Recent logs:"
tail -5 api.log 2>/dev/null || echo "No logs yet"

echo "🎉 Setup completed! API should be running on http://localhost:8000"
echo "📖 Check api.log for detailed logs" 