#!/bin/bash

# Clean RunPod Setup Script

echo "Setting up Voice Cloning API on RunPod..."

# Clean temp directories
rm -rf /tmp/* /logs/* 2>/dev/null || true
chmod 1777 /tmp 2>/dev/null || true
export TMPDIR=/tmp

# Find project directory
if [ -f "requirements.txt" ]; then
    echo "Found requirements.txt in current directory"
elif [ -f "voice-clone-backend/requirements.txt" ]; then
    cd voice-clone-backend
elif [ -f "/workspace/voice-clone-backend/requirements.txt" ]; then
    cd /workspace/voice-clone-backend
else
    echo "Could not find requirements.txt file"
    exit 1
fi

echo "Working directory: $(pwd)"

# Install system dependencies
echo "Installing system dependencies..."
export DEBIAN_FRONTEND=noninteractive

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
    redis-server

# Check GPU
echo "Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "No GPU detected, running on CPU"
fi

# Create directories
mkdir -p ./tmp ./logs
chmod 755 ./tmp ./logs

# Setup Python environment
echo "Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Setup Fish Speech
echo "Setting up Fish Speech..."
if [ ! -d "fish-speech" ]; then
    git clone https://github.com/fishaudio/fish-speech.git
fi

cd fish-speech
pip install -e . --no-deps
cd ..

export PYTHONPATH="${PWD}/fish-speech:${PYTHONPATH}"

# Verify installation
echo "Verifying installation..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Download models if HF_TOKEN provided
if [ ! -z "${HF_TOKEN}" ]; then
    echo "Setting up Hugging Face..."
    pip install huggingface_hub
    echo "${HF_TOKEN}" | huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential || true
    
    mkdir -p checkpoints
    huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini || true
fi

# Stop existing processes
echo "Stopping existing processes..."
pkill -f "python.*main.py" 2>/dev/null || true
pkill -f "rq.*worker" 2>/dev/null || true
pkill -f "workers_starter.py" 2>/dev/null || true
pkill -f "redis-server" 2>/dev/null || true
sleep 3

# Force clean workers
echo "Force cleaning all workers..."
python cleanup_workers.py 2>/dev/null || echo "Cleanup script not available"

# Start Redis with better configuration
echo "Starting Redis server..."
# Clean any leftover Redis data
rm -f dump.rdb 2>/dev/null || true

# Start Redis with proper config
redis-server --daemonize yes --port 6379 --bind 127.0.0.1 --save "" --appendonly no
sleep 2

# Verify Redis with retry
REDIS_RETRIES=5
for i in $(seq 1 $REDIS_RETRIES); do
    if redis-cli ping > /dev/null 2>&1; then
        echo "✅ Redis server running successfully"
        break
    else
        echo "⚠️ Redis attempt $i/$REDIS_RETRIES failed, retrying..."
        sleep 2
        if [ $i -eq $REDIS_RETRIES ]; then
            echo "❌ Failed to start Redis after $REDIS_RETRIES attempts"
            exit 1
        fi
    fi
done

# Start API
echo "Starting API server..."
source venv/bin/activate
nohup ./venv/bin/python main.py > logs/api.log 2>&1 &

# Give API a moment to start
sleep 3

# Check if API started
if pgrep -f "python.*main.py" > /dev/null; then
    echo "✅ API server started successfully"
else
    echo "⚠️ API server may not have started properly, check logs/api.log"
fi

# Start workers with comprehensive setup
echo "Starting RQ Workers..."
mkdir -p logs

# All workers use common log file for easier monitoring  
COMMON_LOG="logs/workers.log"
rm -f "$COMMON_LOG" 2>/dev/null || true

echo "Starting workers (using common log)..."

echo "Starting separation worker..."
nohup ./venv/bin/python workers_starter.py separation_queue sep_worker_1 redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &

echo "Starting dub workers..."
nohup ./venv/bin/python workers_starter.py dub_queue dub_worker_1 redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &
nohup ./venv/bin/python workers_starter.py dub_queue dub_worker_2 redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &

echo "Starting billing worker..."
nohup ./venv/bin/python workers_starter.py billing_queue billing_worker_1 redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &

echo "Waiting for workers to initialize..."
sleep 5

echo "Checking worker status..."
./venv/bin/python check_workers.py || echo "Worker status check completed"

echo ""
echo "🎉 RunPod setup complete! API is ready."
echo "📊 Monitor workers: tail -f logs/workers.log"
echo "🔍 Check status: ./venv/bin/python check_workers.py"
echo "📈 Queue info: ./venv/bin/rq info -u redis://127.0.0.1:6379"
echo "🌐 API should be accessible on port 8000"