#!/bin/bash

# Clean RunPod Setup Script

echo "Setting up Voice Cloning API on RunPod..."

# Clean temp directories
rm -rf /tmp/* /logs/* 2>/dev/null || true
chmod 1777 /tmp 2>/dev/null || true
export TMPDIR=/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FFMPEG_USE_GPU=0
export CUDA_LAUNCH_BLOCKING=0
export TORCH_BACKENDS_CUDNN_DETERMINISTIC=0

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
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo "GPU Memory Status:"
    nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits
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

# Stop existing processes - COMPREHENSIVE CLEANUP
echo "🧹 Performing comprehensive cleanup..."

# Show current processes before cleanup
echo "📊 Current processes before cleanup:"
ps aux | grep -E "(uvicorn|python.*main|rq.*worker|redis-server)" | grep -v grep | head -10 || echo "  - No relevant processes found"

# Kill API servers (multiple patterns to catch all)
echo "⛔ Stopping API servers..."
pkill -f "uvicorn" 2>/dev/null || true
pkill -f "python.*main" 2>/dev/null || true
pkill -f "fastapi" 2>/dev/null || true
pkill -f "gunicorn" 2>/dev/null || true

# Kill ALL Python processes (aggressive but necessary for clean restart)
echo "🐍 Stopping Python processes..."
pkill -f "python.*worker" 2>/dev/null || true
pkill -f "workers_starter.py" 2>/dev/null || true
pkill -f "rq.*worker" 2>/dev/null || true
# Kill any Python process using significant memory (likely AI models)
ps aux | awk '/python/ && $6 > 1000000 {print $2}' | xargs -r kill -TERM 2>/dev/null || true

# Kill Redis gracefully first, then forcefully
echo "📊 Stopping Redis..."
redis-cli shutdown 2>/dev/null || true
sleep 2
pkill -f "redis-server" 2>/dev/null || true

# Kill processes using critical ports
echo "🔌 Freeing ports..."
fuser -k 8000/tcp 2>/dev/null || true
fuser -k 6379/tcp 2>/dev/null || true

# Wait for graceful shutdown
echo "⏳ Waiting for graceful shutdown..."
sleep 8

# Force clean workers with comprehensive error handling
echo "🔧 Deep cleaning workers and cache..."
if [ -f "cleanup_workers.py" ]; then
    python cleanup_workers.py 2>/dev/null || echo "  - Worker cleanup completed with warnings"
else
    echo "  - Manual worker cleanup"
    # Manual Redis cleanup
    redis-cli flushall 2>/dev/null || true
    redis-cli flushdb 2>/dev/null || true
fi

# FORCE KILL remaining processes (nuclear option)
echo "💥 Force killing remaining processes..."
pkill -9 -f "uvicorn" 2>/dev/null || true
pkill -9 -f "python.*main" 2>/dev/null || true
pkill -9 -f "worker" 2>/dev/null || true
pkill -9 -f "redis-server" 2>/dev/null || true

# Clean any high-memory Python processes
ps aux | awk '/python/ && $6 > 1000000 {print $2}' | xargs -r kill -9 2>/dev/null || true

# Clean temp files and caches
echo "🧽 Cleaning temporary files..."
rm -rf /tmp/tmp* 2>/dev/null || true
rm -rf ./tmp/* 2>/dev/null || true
rm -rf ./logs/*.pid 2>/dev/null || true

# Final verification
echo "✅ Cleanup verification:"
sleep 3
REMAINING=$(ps aux | grep -E "(uvicorn|python.*main|rq.*worker|redis-server)" | grep -v grep | wc -l)
if [ "$REMAINING" -gt 0 ]; then
    echo "⚠️  Warning: $REMAINING processes may still be running"
    ps aux | grep -E "(uvicorn|python.*main|rq.*worker|redis-server)" | grep -v grep | head -5 || true
else
    echo "✅ All target processes cleaned successfully"
fi

# Start Redis with better configuration  
echo "🚀 Starting fresh Redis server..."
# Clean any leftover Redis data
rm -f dump.rdb appendonly.aof 2>/dev/null || true
rm -rf /var/lib/redis/* 2>/dev/null || true

# Start Redis with optimized config for AI workload
echo "  - Configuring Redis for high-performance..."
redis-server --daemonize yes \
  --port 6379 \
  --bind 127.0.0.1 \
  --save "" \
  --appendonly no \
  --maxmemory 2gb \
  --maxmemory-policy allkeys-lru \
  --tcp-keepalive 60 \
  --timeout 300
  
sleep 3

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

# Start API server with enhanced monitoring
echo "🚀 Starting ClearVocals API server..."
source venv/bin/activate

# Environment setup
WORKERS=${WORKERS:-1}
HOST=${HOST:-0.0.0.0} 
PORT=${PORT:-8000}

# Clear old logs
> logs/info.log 2>/dev/null || true

echo "  - Host: ${HOST}:${PORT}"
echo "  - Workers: ${WORKERS}"
echo "  - Log: logs/info.log"

# Start with better process management
nohup ./venv/bin/uvicorn main:app \
  --host ${HOST} \
  --port ${PORT} \
  --workers ${WORKERS} \
  --access-log \
  --log-level info \
  > logs/info.log 2>&1 &

API_PID=$!
echo "  - API PID: $API_PID"

# Enhanced startup verification
echo "⏳ Waiting for API initialization..."
sleep 5

# Multiple checks for API readiness
API_READY=false
for i in {1..10}; do
    if pgrep -f "uvicorn.*main:app" > /dev/null; then
        echo "  ✓ Process check passed ($i/10)"
        if curl -s http://localhost:${PORT}/health/live > /dev/null 2>&1; then
            echo "✅ API server ready and responding!"
            API_READY=true
            break
        else
            echo "  - API process running but not responding yet... ($i/10)"
        fi
    else
        echo "  ✗ API process not found ($i/10)"
    fi
    sleep 2
done

if [ "$API_READY" = false ]; then
    echo "⚠️ API server startup verification failed"
    echo "📋 Recent logs:"
    tail -10 logs/info.log || echo "No logs available"
else
    echo "🎯 API server startup successful"
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

echo "Starting dub worker..."
nohup ./venv/bin/python workers_starter.py dub_queue dub_worker_1 redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &

echo "Starting billing worker..."
nohup ./venv/bin/python workers_starter.py billing_queue billing_worker_1 redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &

echo "Waiting for workers to initialize..."
sleep 3

echo "Checking worker status..."
./venv/bin/python check_workers.py || echo "Worker status check completed"

echo ""
echo "🎉 RunPod setup complete! API is ready."
echo "📊 Monitor workers: tail -f logs/workers.log"
echo "🔍 Check status: ./venv/bin/python check_workers.py"
echo "📈 Queue info: ./venv/bin/rq info -u redis://127.0.0.1:6379"
echo "🌐 API should be accessible on port 8000"
echo ""
echo "🖥️  GPU Monitoring Commands:"
echo "   nvidia-smi -l 2  # Monitor GPU every 2 seconds"
echo "   watch -n 1 'nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits'"
echo ""
echo "📊 Performance Tips:"
echo "   - Monitor GPU memory: nvidia-smi"
echo "   - Check API response: curl http://localhost:8000/health/ready"
echo "   - View pipeline stats: check pipeline metrics in logs"