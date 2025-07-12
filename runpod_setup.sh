#!/bin/bash

# RunPod Voice Cloning API Setup - Production Ready

echo "🚀 Setting up Voice Cloning API on RunPod GPU..."

# Set working directory to the project directory
if [ -f "requirements.txt" ]; then
    echo "📁 Found requirements.txt in current directory"
elif [ -f "voice-clone-backend/requirements.txt" ]; then
    echo "📁 Changing to voice-clone-backend directory"
    cd voice-clone-backend
elif [ -f "/workspace/voice-clone-backend/requirements.txt" ]; then
    echo "📁 Changing to /workspace/voice-clone-backend directory"
    cd /workspace/voice-clone-backend
else
    echo "❌ Could not find requirements.txt file"
    echo "Current directory: $(pwd)"
    echo "Files in current directory:"
    ls -la
    exit 1
fi

echo "📁 Working directory: $(pwd)"

# Install system dependencies with error handling
echo "📦 Installing system dependencies..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -y || { echo "❌ Failed to update packages"; exit 1; }
apt-get install -y ffmpeg libsndfile1 python3-dev python3-pip python3-venv git curl build-essential || { echo "❌ Failed to install dependencies"; exit 1; }
apt-get autoremove -y

# Verify GPU availability
echo "🔍 Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
else
    echo "⚠️  GPU not detected, running on CPU"
fi

# Create directories with proper permissions
echo "📁 Creating directories..."
mkdir -p /tmp/voice_cloning ./logs
chmod 755 /tmp/voice_cloning ./logs

# Setup Python environment
echo "🐍 Setting up Python environment..."
python3 -m venv venv || { echo "❌ Failed to create virtual environment"; exit 1; }
source venv/bin/activate || { echo "❌ Failed to activate virtual environment"; exit 1; }
pip install --upgrade pip || { echo "❌ Failed to upgrade pip"; exit 1; }

# Install Python dependencies
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt || { echo "❌ Failed to install requirements"; exit 1; }

# Verify critical packages
echo "🔍 Verifying critical packages..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" || { echo "❌ PyTorch verification failed"; exit 1; }

# Create .env file
echo "⚙️  Creating configuration..."
cat > .env << EOF
# API Configuration
HOST=0.0.0.0
PORT=8000
CUDA_AVAILABLE=true
TEMP_DIR=/tmp/voice_cloning
LOGS_DIR=./logs

# R2 Storage
R2_ACCESS_KEY_ID=${R2_ACCESS_KEY_ID}
R2_SECRET_ACCESS_KEY=${R2_SECRET_ACCESS_KEY}
R2_BUCKET_NAME=${R2_BUCKET_NAME}
R2_ENDPOINT=${R2_ENDPOINT}
R2_PUBLIC_URL=${R2_PUBLIC_URL}

# API Keys
OPENAI_API_KEY=${OPENAI_API_KEY}
ASSEMBLYAI_API_KEY=${ASSEMBLYAI_API_KEY}
API_ACCESS_TOKEN=${API_ACCESS_TOKEN}
RUNPOD_ENDPOINT_ID=${RUNPOD_ENDPOINT_ID}
RUNPOD_TIMEOUT=${RUNPOD_TIMEOUT:-1800000}

# MongoDB
MONGODB_URI=${MONGODB_URI}

# Processing Options
ENABLE_SUBTITLES=true
ENABLE_INSTRUMENTS=true
EOF

# Kill existing processes
echo "🔄 Stopping existing processes..."
pkill -f "python.*main.py" || true
sleep 3

# Start API with proper virtual environment
echo "🚀 Starting API server..."
source venv/bin/activate
nohup ./venv/bin/python main.py > ./logs/api.log 2>&1 &

# Wait for API to start
echo "⏳ Waiting for API to start..."
sleep 10

echo "🎉 Setup complete! Your Voice Cloning API is ready!" 