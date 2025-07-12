#!/bin/bash

# RunPod Voice Cloning API Setup - Production Ready

echo "🚀 Setting up Voice Cloning API on RunPod GPU..."

# Set working directory
cd /workspace

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
mkdir -p /tmp/voice_cloning /workspace/logs
chmod 755 /tmp/voice_cloning /workspace/logs

# Setup Python environment
echo "🐍 Setting up Python environment..."
if [ -d "venv" ]; then
    rm -rf venv
fi
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
nohup ./venv/bin/python main.py > /workspace/logs/api.log 2>&1 &

# Wait for API to start
echo "⏳ Waiting for API to start..."
sleep 10

# Health check
echo "🔍 Performing health check..."
for i in {1..10}; do
    if curl -f http://localhost:8000/ &>/dev/null; then
        echo "✅ API is running successfully!"
        echo "🌐 API URL: http://0.0.0.0:8000"
        echo "📊 API Status: http://0.0.0.0:8000/"
        echo "📋 Logs: tail -f /workspace/logs/api.log"
        echo "🎯 Health Check: curl http://localhost:8000/"
        break
    else
        echo "⏳ Waiting for API... (attempt $i/10)"
        sleep 5
    fi
    
    if [ $i -eq 10 ]; then
        echo "❌ API failed to start. Check logs:"
        tail -20 /workspace/logs/api.log
        exit 1
    fi
done

echo "🎉 Setup complete! Your Voice Cloning API is ready!" 