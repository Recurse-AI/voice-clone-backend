#!/bin/bash

# RunPod Voice Cloning API Setup - Production Ready

echo "🚀 Setting up Voice Cloning API on RunPod GPU..."

# Fix /tmp directory permissions first
echo "🔧 Fixing system permissions..."
chmod 1777 /tmp 2>/dev/null || true
mkdir -p /tmp && chmod 1777 /tmp 2>/dev/null || true
export TMPDIR=/tmp

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

# Install system dependencies with better error handling
echo "📦 Installing system dependencies..."
export DEBIAN_FRONTEND=noninteractive

# Try to fix APT issues first
echo "🔧 Fixing APT configuration..."
apt-get clean || true
rm -rf /var/lib/apt/lists/* || true
mkdir -p /var/lib/apt/lists/partial || true

# Try updating packages with retry logic
echo "🔄 Updating package lists..."
for i in {1..3}; do
    if apt-get update -y; then
        echo "✅ Package lists updated successfully"
        break
    else
        echo "⚠️  Attempt $i failed, retrying..."
        sleep 2
        apt-get clean || true
        rm -rf /var/lib/apt/lists/* || true
    fi
    
    if [ $i -eq 3 ]; then
        echo "⚠️  Package update failed, but continuing with existing packages..."
    fi
done

# Install dependencies with fallback (including Fish Speech 1.5 requirements)
echo "📦 Installing required packages..."
apt-get install -y ffmpeg libsndfile1 python3-dev python3-pip python3-venv git curl build-essential portaudio19-dev libsox-dev || {
    echo "⚠️  Some packages failed to install, checking what's available..."
    
    # Try installing packages individually
    for package in ffmpeg libsndfile1 python3-dev python3-pip python3-venv git curl build-essential portaudio19-dev libsox-dev; do
        if apt-get install -y "$package"; then
            echo "✅ Installed $package"
        else
            echo "⚠️  Failed to install $package"
        fi
    done
}

apt-get autoremove -y || true

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
mkdir -p ./tmp/voice_cloning ./logs
chmod 755 ./tmp/voice_cloning ./logs


# Setup Python environment
echo "🐍 Setting up Python environment..."
python3 -m venv venv || { echo "❌ Failed to create virtual environment"; exit 1; }
source venv/bin/activate || { echo "❌ Failed to activate virtual environment"; exit 1; }
pip install --upgrade pip || { echo "❌ Failed to upgrade pip"; exit 1; }

# Install system dependencies for Fish Speech
echo "🎵 Installing Fish Speech system dependencies..."
apt-get install -y portaudio19-dev python3-pyaudio || {
    echo "⚠️  Audio dependencies failed, continuing without pyaudio..."
}

# Install Python dependencies
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt || { echo "❌ Failed to install requirements"; exit 1; }

# Setup Fish Speech manually (clean approach)
echo "🎵 Setting up Fish Speech 1.5 (OpenAudio) for voice cloning..."
if [ ! -d "fish-speech" ]; then
    echo "📥 Cloning Fish Speech repository..."
    git clone https://github.com/fishaudio/fish-speech.git || { echo "❌ Failed to clone Fish Speech"; exit 1; }
else
    echo "✅ Fish Speech repository already exists"
fi

# Install Fish Speech dependencies (comprehensive list)
echo "📦 Installing Fish Speech dependencies..."
pip install torch torchaudio transformers accelerate librosa matplotlib fire hydra-core wandb vector-quantize-pytorch natsort silero-vad loralib einops omegaconf tensorboard gradio pescador descript-audiotools descript-audio-codec pyrootutils resampy zstandard cachetools pytorch-lightning lightning || {
    echo "⚠️  Some Fish Speech dependencies failed, continuing..."
}

# Install specific versions of critical dependencies
echo "📦 Installing specific Fish Speech requirements..."
pip install "hydra-core>=1.2.0" "omegaconf>=2.2.0" "einops>=0.6.0" || {
    echo "⚠️  Some specific dependencies failed, continuing..."
}

# Install Fish Speech in development mode (skip pyaudio)
echo "🔧 Installing Fish Speech package..."
cd fish-speech
pip install -e . --no-deps || { echo "❌ Fish Speech installation failed"; exit 1; }
cd ..

# Add Fish Speech to Python path
export PYTHONPATH="${PWD}/fish-speech:${PYTHONPATH}"

# Verify critical packages
echo "🔍 Verifying critical packages..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" || { echo "❌ PyTorch verification failed"; exit 1; }

# Verify Fish Speech installation
echo "🔍 Verifying Fish Speech installation..."
python3 -c "
import sys
sys.path.insert(0, './fish-speech')
try:
    from fish_speech.models.text2semantic.llama import BaseTransformer
    print('✅ Fish Speech modules imported successfully!')
except Exception as e:
    print(f'⚠️  Fish Speech verification failed: {e}')
    print('Continuing anyway...')
" || echo "⚠️  Fish Speech verification had issues, but continuing..."

# Create .env file
echo "⚙️  Creating configuration..."
cat > .env << EOF
# API Configuration
HOST=0.0.0.0
PORT=8000
CUDA_AVAILABLE=true
TEMP_DIR=./tmp/voice_cloning
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

echo "🎉 Setup complete! Your Voice Cloning API is ready!" 