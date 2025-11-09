#!/bin/bash

rm -rf logs/* tmp/*

echo "🚀 Starting Voice Cloning API on RunPod..."

apt-get update
apt-get install -y ffmpeg

apt-get install -y libcudnn8-dev libcudnn8 

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FFMPEG_USE_GPU=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_HOME=/usr
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/bin:/usr/local/cuda/bin:$PATH
export FORCE_CUDA=1

echo "🐍 Activating virtual environment..."
source venv/bin/activate

# Check if fish-speech exists, clone if missing
if [ ! -d "fish-speech" ]; then
    echo "🐟 Fish Speech not found, cloning..."
    git clone https://github.com/fishaudio/fish-speech.git
    cd fish-speech
    pip install -e . --no-deps
    cd ..
    export PYTHONPATH="${PWD}/fish-speech:${PYTHONPATH}"
else
    echo "✅ Fish Speech already exists, skipping clone"
fi

# Check and update yt-dlp if newer version available
echo "🔄 Checking yt-dlp version..."
CURRENT_VERSION=$(pip show yt-dlp 2>/dev/null | grep "Version:" | awk '{print $2}' || echo "not_installed")
if [ "$CURRENT_VERSION" = "not_installed" ]; then
    echo "📦 Installing yt-dlp..."
    pip install yt-dlp --quiet
else
    echo "📋 Current yt-dlp version: $CURRENT_VERSION"
    echo "🔄 Updating yt-dlp to latest version..."
    pip install --upgrade yt-dlp --quiet
    NEW_VERSION=$(pip show yt-dlp 2>/dev/null | grep "Version:" | awk '{print $2}')
    if [ "$CURRENT_VERSION" != "$NEW_VERSION" ]; then
        echo "✅ Updated yt-dlp: $CURRENT_VERSION → $NEW_VERSION"
    else
        echo "✅ yt-dlp already at latest version ($CURRENT_VERSION)"
    fi
fi

# Check and install cuDNN if missing (required for WhisperX/PyAnnote)
echo "🔍 Checking GPU dependencies..."
if command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1)
    
    # Check if cuDNN is already installed
    python -c "import nvidia.cudnn" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "📦 Installing cuDNN for CUDA ${CUDA_VERSION}..."
        if [ "$CUDA_VERSION" = "12" ] || [ "$CUDA_VERSION" -ge 12 ]; then
            pip install --no-cache-dir nvidia-cudnn-cu12 -q
        else
            pip install --no-cache-dir nvidia-cudnn-cu11 -q
        fi
        echo "✅ cuDNN installed successfully"
    else
        echo "✅ cuDNN already installed"
    fi  
else
    echo "⚠️ No GPU detected, skipping cuDNN installation"
fi

#clear all workers
python cleanup_workers.py


mkdir -p ./tmp ./logs
chmod 755 ./tmp ./logs

echo "🧹 Cleanup previous processes..."
pkill -9 -f "uvicorn.*main:app" 2>/dev/null || true
pkill -9 -f "workers_starter.py" 2>/dev/null || true

rm -rf ./tmp/* ./logs/*.pid ./logs/*.log 2>/dev/null || true
rm -f dump.rdb appendonly.aof *.rdb *.aof 2>/dev/null || true

if command -v nvidia-smi >/dev/null 2>&1; then
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
fi

sleep 1


echo "🚀 Starting Redis..."
rm -f dump.rdb appendonly.aof 2>/dev/null || true
redis-server --daemonize yes --port 6379 --bind 127.0.0.1 --save "" --appendonly no --maxmemory 2gb --maxmemory-policy allkeys-lru

sleep 2

for i in 1 2 3; do
    if redis-cli ping > /dev/null 2>&1; then
        echo "✅ Redis running"
        break
    fi
    sleep 1
done

WORKERS=${WORKERS:-4}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}

echo "🧹 Killing process on port ${PORT}..."
lsof -ti tcp:${PORT} | xargs kill -9 2>/dev/null || fuser -k ${PORT}/tcp 2>/dev/null || true

echo "🚀 Starting API server on port ${PORT}..."

> logs/info.log 2>/dev/null || true

export OMP_NUM_THREADS=8
export TORCH_NUM_THREADS=4
export MKL_NUM_THREADS=8
export LOAD_AI_MODELS=false

nohup ./venv/bin/uvicorn main:app --host ${HOST} --port ${PORT} --workers ${WORKERS} --access-log --log-level info > logs/info.log 2>&1 &

sleep 3

if pgrep -f "uvicorn.*main:app" > /dev/null; then
    echo "✅ API server running"
fi

echo "🔧 Starting RQ Workers..."
mkdir -p logs
COMMON_LOG="logs/workers.log"
rm -f "$COMMON_LOG" 2>/dev/null || true

SEPARATION_WORKERS=${MAX_SEPARATION_WORKERS:-2}
for i in $(seq 1 $SEPARATION_WORKERS); do
    nohup ./venv/bin/python workers_starter.py separation_queue sep_worker_${i} redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &
    sleep 1
done

DUB_WORKERS=${MAX_DUB_ORCHESTRATION_WORKERS:-4}
for i in $(seq 1 $DUB_WORKERS); do
    nohup ./venv/bin/python workers_starter.py dub_queue dub_worker_${i} redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &
    sleep 1
done

nohup ./venv/bin/python workers_starter.py dub_queue resume_worker_1 redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &
sleep 1
nohup ./venv/bin/python workers_starter.py dub_queue resume_worker_2 redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &

nohup ./venv/bin/python workers_starter.py billing_queue billing_worker_1 redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &

CLIP_WORKERS=${MAX_CLIP_WORKERS:-2}
for i in $(seq 1 $CLIP_WORKERS); do
    nohup ./venv/bin/python workers_starter.py clip_queue clip_worker_${i} redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &
    sleep 1
done

nohup ./venv/bin/python workers_starter.py whisperx_service_queue whisperx_service_worker_1 redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &
nohup ./venv/bin/python workers_starter.py fish_speech_service_queue fish_speech_service_worker_1 redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &


VIDEO_WORKERS=${MAX_VIDEO_PROCESSING_WORKERS:-2}
for i in $(seq 1 $VIDEO_WORKERS); do
    nohup ./venv/bin/python workers_starter.py video_processing_queue video_worker_${i} redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &
    sleep 1
done

sleep 10

./venv/bin/python check_workers.py || echo "Worker status check completed"

echo ""
echo "🎉 RunPod setup complete!"
echo ""
echo "📊 Monitor: tail -f logs/workers.log"
echo "🌐 API: http://localhost:${PORT}/"