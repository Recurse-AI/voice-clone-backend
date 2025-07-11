#!/bin/bash

# RunPod Voice Cloning API Setup - Essential Only

echo "🚀 Setting up Voice Cloning API..."

# Install system dependencies
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y ffmpeg libsndfile1 python3-dev python3-pip python3-venv git curl
apt-get autoremove -y

# Create directories
mkdir -p /tmp/voice_cloning /workspace/logs

# Setup Python environment
if [ -d "venv" ]; then
    rm -rf venv
fi
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Create .env file
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
pkill -f "python.*main.py" || true
sleep 2

# Start API
echo "🚀 Starting API..."
nohup python3 main.py > /workspace/logs/api.log 2>&1 &

echo "✅ Setup complete! API running on http://0.0.0.0:8000"
echo "📋 Logs: tail -f /workspace/logs/api.log" 