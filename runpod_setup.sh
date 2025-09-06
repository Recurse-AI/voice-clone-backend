#!/bin/bash

# Simplified RunPod Startup Script
# Use environment_setup.sh first for initial setup

echo "🚀 Starting Voice Cloning API on RunPod..."

# Basic environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FFMPEG_USE_GPU=0
export CUDA_LAUNCH_BLOCKING=0
export TORCH_BACKENDS_CUDNN_DETERMINISTIC=0



# Activate virtual environment
echo "🐍 Activating virtual environment..."
source venv/bin/activate

# Create directories
mkdir -p ./tmp ./logs
chmod 755 ./tmp ./logs
# COMPREHENSIVE CLEANUP
echo "🧹 Performing comprehensive cleanup..."

# Show current processes
echo "📊 Current processes before cleanup:"
ps aux | grep -E "(uvicorn|python.*main|rq.*worker|redis-server)" | grep -v grep | head -10 || echo "  - No relevant processes found"

# Kill processes
echo "⛔ Stopping existing processes..."
pkill -f "uvicorn" 2>/dev/null || true
pkill -f "python.*main" 2>/dev/null || true
pkill -f "fastapi" 2>/dev/null || true
pkill -f "gunicorn" 2>/dev/null || true
pkill -f "python.*worker" 2>/dev/null || true
pkill -f "workers_starter.py" 2>/dev/null || true
pkill -f "rq.*worker" 2>/dev/null || true

# Kill Redis gracefully
echo "📊 Stopping Redis..."
redis-cli shutdown 2>/dev/null || true
sleep 2
pkill -f "redis-server" 2>/dev/null || true

# Free ports
echo "🔌 Freeing ports..."
fuser -k 8000/tcp 2>/dev/null || true
fuser -k 6379/tcp 2>/dev/null || true

# Wait for cleanup
echo "⏳ Waiting for graceful shutdown..."
sleep 5

# Force cleanup workers
echo "🔧 Cleaning workers..."
if [ -f "cleanup_workers.py" ]; then
    python cleanup_workers.py 2>/dev/null || echo "  - Worker cleanup completed"
fi

# Force kill remaining processes
echo "💥 Force killing remaining processes..."
pkill -9 -f "uvicorn" 2>/dev/null || true
pkill -9 -f "python.*main" 2>/dev/null || true
pkill -9 -f "worker" 2>/dev/null || true
pkill -9 -f "redis-server" 2>/dev/null || true

# Clean temp files
echo "🧽 Cleaning temporary files..."
rm -rf /tmp/tmp* 2>/dev/null || true
rm -rf ./tmp/* 2>/dev/null || true
rm -rf ./logs/*.pid 2>/dev/null || true

# Cleanup verification
echo "✅ Cleanup verification:"
sleep 2
REMAINING=$(ps aux | grep -E "(uvicorn|python.*main|rq.*worker|redis-server)" | grep -v grep | wc -l)
if [ "$REMAINING" -gt 0 ]; then
    echo "⚠️  Warning: $REMAINING processes may still be running"
else
    echo "✅ All target processes cleaned successfully"
fi

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
  --tcp-keepalive 60 \
  --timeout 300

sleep 3

# Verify Redis
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

# START API SERVER
echo "🚀 Starting ClearVocals API server..."

# Server configuration
WORKERS=${WORKERS:-10}
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

echo "🚀 Starting API server with ${WORKERS} lightweight workers..."
nohup ./venv/bin/uvicorn main:app \
  --host ${HOST} \
  --port ${PORT} \
  --workers ${WORKERS} \
  --access-log \
  --log-level info \
  > logs/info.log 2>&1 &

API_PID=$!
echo "✅ API started with ${WORKERS} workers, PID: $API_PID"

# API verification
echo "⏳ Initializing API..."
sleep 8

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

echo "  - Starting separation worker..."
nohup ./venv/bin/python workers_starter.py separation_queue sep_worker_1 redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &

echo "  - Starting dub worker with AI models..."
LOAD_AI_MODELS=true nohup ./venv/bin/python workers_starter.py dub_queue dub_worker_1 redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &

echo "  - Starting billing worker..."
nohup ./venv/bin/python workers_starter.py billing_queue billing_worker_1 redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &

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
