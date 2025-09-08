#!/bin/bash

# Simplified RunPod Startup Script
# Use environment_setup.sh first for initial setup

echo "🚀 Starting Voice Cloning API on RunPod..."

# Basic environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FFMPEG_USE_GPU=0
export CUDA_LAUNCH_BLOCKING=0
export TORCH_BACKENDS_CUDNN_DETERMINISTIC=0

# GPU Configuration - Force CUDA usage
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID  
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"

# CUDA Runtime Environment
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH

# Force GPU detection 
export FORCE_CUDA=1

# AI Model optimization - Disabled compilation for faster processing
export FISH_SPEECH_COMPILE=false  # Disabled for instant voice cloning
export TORCH_JIT_LOG_LEVEL=ERROR
export TORCH_COMPILE_MODE=reduce-overhead
export TORCH_COMPILE_BACKEND=inductor



# Activate virtual environment
echo "🐍 Activating virtual environment..."
source venv/bin/activate

# Create directories
mkdir -p ./tmp ./logs
chmod 755 ./tmp ./logs
# FAST CLEANUP
echo "🧹 Quick cleanup..."

# Kill processes efficiently
pkill -f "uvicorn" 2>/dev/null || true
pkill -f "python.*main" 2>/dev/null || true
pkill -f "python.*worker" 2>/dev/null || true
pkill -f "rq.*worker" 2>/dev/null || true
redis-cli shutdown 2>/dev/null || true
pkill -f "redis-server" 2>/dev/null || true

# Free ports
fuser -k 8000/tcp 2>/dev/null || true
fuser -k 6379/tcp 2>/dev/null || true

# Quick cleanup
rm -rf ./tmp/* ./logs/*.pid 2>/dev/null || true

echo "✅ Fast cleanup completed"

# START REDIS
echo "🚀 Starting fresh Redis server..."
rm -f dump.rdb appendonly.aof 2>/dev/null || true

redis-server --daemonize yes \
  --port 6379 \
  --bind 127.0.0.1 \
  --save "" \
  --appendonly no \
  --maxmemory 2gb \
  --maxmemory-policy allkeys-lru \
  --tcp-keepalive 300 \
  --timeout 0 \
  --maxclients 10000 \
  --tcp-backlog 1024

sleep 2

# Quick Redis verification
REDIS_RETRIES=3
for i in $(seq 1 $REDIS_RETRIES); do
    if redis-cli ping > /dev/null 2>&1; then
        echo "✅ Redis server running"
        break
    else
        sleep 1
        if [ $i -eq $REDIS_RETRIES ]; then
            echo "❌ Redis startup failed"
            exit 1
        fi
    fi
done

# START API SERVER
echo "🚀 Starting ClearVocals API server..."

# Server configuration - Optimized for RunPod
WORKERS=${WORKERS:-4}  # Reduced from 10 to 4 for efficiency
HOST=${HOST:-0.0.0.0} 
PORT=${PORT:-8000}

# Clear old logs
> logs/info.log 2>/dev/null || true

echo "  - Host: ${HOST}:${PORT}"
echo "  - Workers: ${WORKERS}"
echo "  - Log: logs/info.log"

# Environment for lightweight API workers
export OMP_NUM_THREADS=8
export TORCH_NUM_THREADS=4
export MKL_NUM_THREADS=8
export LOAD_AI_MODELS=false

echo "🚀 Starting API server with ${WORKERS} optimized workers..."
nohup ./venv/bin/uvicorn main:app \
  --host ${HOST} \
  --port ${PORT} \
  --workers ${WORKERS} \
  --access-log \
  --log-level info \
  > logs/info.log 2>&1 &

API_PID=$!
echo "✅ API started with ${WORKERS} workers, PID: $API_PID"

# Quick API verification
echo "⏳ Verifying API startup..."
sleep 3

if pgrep -f "uvicorn.*main:app" > /dev/null; then
    echo "✅ API server running"
else
    echo "⚠️ API startup failed, check logs/info.log"
fi

# START WORKERS
echo "🔧 Starting RQ Workers..."
mkdir -p logs

COMMON_LOG="logs/workers.log"
rm -f "$COMMON_LOG" 2>/dev/null || true

echo "Starting workers..."

echo "🔍 Setting up separation workers..."
SEPARATION_WORKERS=${MAX_SEPARATION_WORKERS:-2}

echo "  - Starting ${SEPARATION_WORKERS} separation worker(s)..."
for i in $(seq 1 $SEPARATION_WORKERS); do
    echo "    - Starting sep_worker_${i}..."
    LOAD_WHISPERX_MODEL=false LOAD_FISH_SPEECH_MODEL=false nohup ./venv/bin/python workers_starter.py separation_queue sep_worker_${i} redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &
    sleep 1
done

# Dub orchestration workers (VRAM managed by service workers)
echo "🔍 Setting up dub orchestration workers..."
DUB_WORKERS=${MAX_DUB_ORCHESTRATION_WORKERS:-4}

if command -v nvidia-smi >/dev/null 2>&1; then
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    VRAM_GB=$((VRAM_MB / 1024))
    echo "  - GPU: ${VRAM_GB}GB VRAM detected"
else
    echo "  - No GPU detected"
fi

echo "  - Starting ${DUB_WORKERS} dub orchestration workers (VRAM managed by service workers)"

echo "  - Starting ${DUB_WORKERS} dub orchestration worker(s) (no AI models)..."
for i in $(seq 1 $DUB_WORKERS); do
    echo "    - Starting dub_worker_${i}..."
    LOAD_WHISPERX_MODEL=false LOAD_FISH_SPEECH_MODEL=false nohup ./venv/bin/python workers_starter.py dub_queue dub_worker_${i} redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &
    sleep 1  # Quick stagger for clean startup
done

echo "  - Starting 2 dedicated RESUME workers for instant job resumption..."
LOAD_WHISPERX_MODEL=false LOAD_FISH_SPEECH_MODEL=false nohup ./venv/bin/python workers_starter.py dub_queue resume_worker_1 redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &
sleep 1
LOAD_WHISPERX_MODEL=false LOAD_FISH_SPEECH_MODEL=false nohup ./venv/bin/python workers_starter.py dub_queue resume_worker_2 redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &

echo "  - Starting billing worker..."
nohup ./venv/bin/python workers_starter.py billing_queue billing_worker_1 redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &

# VRAM Service Workers (Serial Processing)
echo "🎯 Starting VRAM service workers..."

echo "  - Starting WhisperX service worker (1 worker for 16GB VRAM)..."
LOAD_WHISPERX_MODEL=true LOAD_FISH_SPEECH_MODEL=false nohup ./venv/bin/python workers_starter.py whisperx_service_queue whisperx_service_worker_1 redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &

echo "  - Starting Fish Speech service worker (1 worker for 16GB VRAM)..."
LOAD_WHISPERX_MODEL=false LOAD_FISH_SPEECH_MODEL=true nohup ./venv/bin/python workers_starter.py fish_speech_service_queue fish_speech_service_worker_1 redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &

echo "⏳ Waiting for VRAM workers to load models..."
sleep 10

echo "⏳ Waiting for workers to initialize..."
sleep 5

echo "📊 Checking worker status..."
./venv/bin/python check_workers.py || echo "Worker status check completed"

echo ""
echo "🎉 RunPod setup complete! API is ready."
echo ""
echo "📊 Monitor Commands:"
echo "   tail -f logs/workers.log              # Monitor workers"
echo "   tail -f logs/info.log                 # Monitor API"
echo "   ./venv/bin/python check_workers.py    # Check status"
echo "   ./venv/bin/rq info -u redis://127.0.0.1:6379  # Queue info"
echo ""
echo "🌐 API Endpoints:"
echo "   http://localhost:${PORT}/              # Status"
echo "   http://localhost:${PORT}/health/ready  # Health check"
echo ""
echo "🖥️  GPU Monitoring:"
echo "   nvidia-smi -l 2    # Monitor GPU every 2 seconds"
echo ""
echo "🔴 Stop Commands:"
echo "   pkill -f 'uvicorn.*main:app' && pkill -f 'rq.*worker'"
