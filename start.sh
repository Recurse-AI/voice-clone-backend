#!/bin/bash

# Voice Cloning API Startup Script
echo "🚀 Starting Voice Cloning API..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found. Please create one based on .env.example"
    exit 1
fi

# Create temp directory
mkdir -p /tmp/voice_cloning
echo "📁 Created temporary directory"

# Install dependencies if needed
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
fi

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "🖥️  GPU detected, enabling CUDA"
    export CUDA_AVAILABLE=true
else
    echo "🖥️  No GPU detected, using CPU"
    export CUDA_AVAILABLE=false
fi

# Start the API
echo "🎵 Starting Voice Cloning API on port 8000..."
python main.py 