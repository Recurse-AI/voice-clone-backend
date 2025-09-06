#!/bin/bash

# Clean API Startup Script
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
export TORCH_BACKENDS_CUDNN_DETERMINISTIC=0
export FFMPEG_USE_GPU=0

echo "Activating virtual environment..."
# Windows and Linux compatible venv activation
if [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
fi

echo "Installing dependencies..."
pip install --upgrade pip
# pip install -r requirements.txt || echo "Failed to install some dependencies, continuing..."


echo "Setting up Redis for WSL Ubuntu..."
echo "Stopping existing processes..."
# Kill uvicorn/main and any lingering rq workers
pkill -f "python.*main.py" 2>/dev/null || true
pkill -f "rq.*worker" 2>/dev/null || true
pkill -f "workers_starter.py" 2>/dev/null || true

echo "Force cleaning all workers..."
python cleanup_workers.py
sleep 2

echo "Using WSL Redis (assuming running)..."
# WSL Ubuntu Redis is managed separately

sleep 3
mkdir -p logs

echo "Testing WSL Redis connection..."
# Test Redis connection with Python since redis-cli might not be in PATH
python -c "
import redis
try:
    r = redis.Redis(host='127.0.0.1', port=6379, db=0)
    r.ping()
    print('✅ WSL Redis connection successful')
except Exception as e:
    print(f'❌ Redis connection failed: {e}')
    print('Make sure Redis is running in WSL: sudo service redis-server start')
"

echo "Starting RQ Workers..."
COMMON_LOG="logs/workers.log"
rm -f "$COMMON_LOG" 2>/dev/null || true

echo "Starting separation worker..."
nohup python workers_starter.py separation_queue sep_worker_1 redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &

echo "Starting dub worker..."
nohup python workers_starter.py dub_queue dub_worker_1 redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &

echo "Starting billing worker..."
nohup python workers_starter.py billing_queue billing_worker_1 redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &

echo "Waiting for workers to initialize..."
sleep 5

echo "Checking worker status..."
python check_workers.py

echo ""
echo "Waiting additional time for workers to fully start..."
sleep 3

echo "Final worker status check..."
python check_workers.py

echo "Checking worker logs for errors..."
if [ -f "logs/workers.log" ]; then
    echo "Last 10 lines of worker log:"
    tail -10 logs/workers.log
else
    echo "No worker log file found"
fi

echo ""
echo "🎉 WSL Ubuntu Setup Complete!"
echo ""
echo "📊 Monitor Commands:"
echo "   tail -f logs/workers.log    # Monitor workers"
echo "   python check_workers.py     # Check worker status"
echo ""
echo "🔴 Stop Commands:"
echo "   pkill -f 'python.*main.py' && pkill -f 'rq.*worker'"
echo ""
echo "🔧 WSL Redis Commands:"
echo "   sudo service redis-server start    # Start Redis"
echo "   sudo service redis-server stop     # Stop Redis"
echo "   sudo service redis-server status   # Check Redis"
echo ""
echo "Starting API server..."

# Start main API
python main.py

echo "Services started successfully!"