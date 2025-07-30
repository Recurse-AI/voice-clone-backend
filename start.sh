#!/bin/bash

# Local Voice Cloning API Startup Script

echo "🚀 Starting Voice Cloning API locally..."

# Set working directory to the project directory
echo "📁 Working directory: $(pwd)"

# Load environment variables from .env file
if [ -f ".env" ]; then
    echo "📝 Loading environment variables from .env file..."
    export $(grep -v '^#' .env | xargs)
    echo "✅ Environment variables loaded"
else
    echo "❌ .env file not found!"
    exit 1
fi

# Create temp and logs directories
echo "📁 Creating directories..."
mkdir -p ./tmp/voice_cloning ./logs
chmod 755 ./tmp/voice_cloning ./logs

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "🐍 Creating Python virtual environment..."
    if command -v python3 &> /dev/null; then
        python3 -m venv venv || { echo "❌ Failed to create virtual environment"; exit 1; }
    elif command -v python &> /dev/null; then
        python -m venv venv || { echo "❌ Failed to create virtual environment"; exit 1; }
    else
        echo "❌ Python not found!"
        exit 1
    fi
fi

# Activate virtual environment
echo "🐍 Activating virtual environment..."
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate || { echo "❌ Failed to activate virtual environment"; exit 1; }
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate || { echo "❌ Failed to activate virtual environment"; exit 1; }
else
    echo "❌ Virtual environment activation script not found!"
    exit 1
fi

# Upgrade pip
echo "📦 Upgrading pip..."
python -m pip install --upgrade pip || {
    echo "⚠️  Pip upgrade failed, continuing with existing version..."
}

# Install requirements if needed
if [ -f "requirements.txt" ]; then
    echo "📚 Checking Python dependencies..."
    pip install -r requirements.txt || { echo "❌ Failed to install requirements"; exit 1; }
else
    echo "⚠️  requirements.txt not found, skipping dependency installation"
fi

# Setup Fish Speech if not already exists
if [ ! -d "fish-speech" ]; then
    echo "🎵 Setting up Fish Speech for voice cloning..."
    echo "📥 Cloning Fish Speech repository..."
    git clone https://github.com/fishaudio/fish-speech.git || { echo "❌ Failed to clone Fish Speech"; exit 1; }
    
    echo "📦 Installing Fish Speech dependencies..."
    pip install torch torchaudio transformers accelerate librosa matplotlib fire hydra-core wandb vector-quantize-pytorch natsort silero-vad loralib einops omegaconf tensorboard gradio pescador descript-audiotools descript-audio-codec pyrootutils resampy zstandard cachetools pytorch-lightning lightning || {
        echo "⚠️  Some Fish Speech dependencies failed, continuing..."
    }
    
    echo "🔧 Installing Fish Speech package..."
    cd fish-speech
    pip install -e . --no-deps || { echo "❌ Fish Speech installation failed"; exit 1; }
    cd ..
else
    echo "✅ Fish Speech already exists"
fi

# Add Fish Speech to Python path
export PYTHONPATH="${PWD}/fish-speech:${PYTHONPATH}"

# Verify critical packages
echo "🔍 Verifying critical packages..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" || { echo "❌ PyTorch verification failed"; exit 1; }

# Setup models if HF_TOKEN is available
if [ ! -z "${HF_TOKEN}" ]; then
    echo "🔑 HF_TOKEN found, checking models..."
    pip install huggingface_hub || { echo "❌ Failed to install huggingface_hub"; exit 1; }
    
    if [ ! -d "checkpoints/openaudio-s1-mini" ]; then
        echo "📥 Downloading OpenAudio S1-mini models..."
        mkdir -p checkpoints
        huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini || {
            echo "⚠️ Model download failed, will download during runtime"
        }
    else
        echo "✅ Models already downloaded"
    fi
else
    echo "⚠️ HF_TOKEN not found, models will be downloaded during runtime if needed"
fi

# Kill existing processes
echo "🔄 Stopping existing processes..."
pkill -f "python.*main.py" || true
sleep 2

# Start API
echo "🚀 Starting API server..."
python main.py

echo "🎉 API server started!"
