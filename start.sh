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


echo ""
echo "🛠️ Setting up Redis for WSL Ubuntu..."

echo "🧹 Stopping existing processes..."

# Cross-platform process cleanup
if command -v wmic >/dev/null 2>&1; then
    # Windows system detected
    echo "   - Windows detected, using wmic/taskkill..."
    
    # Kill Python processes (main.py, workers_starter.py)
    wmic process where "name='python.exe' and commandline like '%main.py%'" delete 2>/dev/null || true
    wmic process where "name='python.exe' and commandline like '%workers_starter.py%'" delete 2>/dev/null || true
    wmic process where "name='python.exe' and commandline like '%uvicorn%'" delete 2>/dev/null || true
    
    # Alternative taskkill approach
    taskkill /F /IM python.exe /FI "WINDOWTITLE eq *main.py*" 2>/dev/null || true
    taskkill /F /IM python.exe /FI "WINDOWTITLE eq *worker*" 2>/dev/null || true
    
elif command -v pkill >/dev/null 2>&1; then
    # Unix/Linux/WSL system detected
    echo "   - Unix/Linux detected, using pkill..."
    pkill -f "python.*main.py" 2>/dev/null || true
    pkill -f "uvicorn.*main:app" 2>/dev/null || true
    pkill -f "rq.*worker" 2>/dev/null || true
    pkill -f "workers_starter.py" 2>/dev/null || true
    pkill -f "python.*video_processing_worker" 2>/dev/null || true
else
    # Fallback: try both approaches
    echo "   - Unknown system, trying both approaches..."
    pkill -f "python.*main.py" 2>/dev/null || true
    pkill -f "rq.*worker" 2>/dev/null || true
    taskkill /F /IM python.exe 2>/dev/null || true
fi

echo "🧹 Force cleaning all workers..."
python cleanup_workers.py 2>/dev/null || true
sleep 3

# Cleanup verification (cross-platform)
echo "✅ Cleanup verification:"
if command -v wmic >/dev/null 2>&1; then
    # Windows verification
    REMAINING=$(wmic process where "name='python.exe' and (commandline like '%main.py%' or commandline like '%worker%')" get processid 2>/dev/null | wc -l 2>/dev/null || echo "0")
elif command -v ps >/dev/null 2>&1; then
    # Unix/Linux verification
    REMAINING=$(ps aux 2>/dev/null | grep -E "(uvicorn|python.*main|rq.*worker)" | grep -v grep | wc -l 2>/dev/null || echo "0")
else
    REMAINING="unknown"
fi

if [ "$REMAINING" = "unknown" ]; then
    echo "⚠️  Cannot verify cleanup - system commands not available"
    echo "   Manually check for running Python processes if needed"
elif [ "$REMAINING" -gt 0 ]; then
    echo "⚠️  Warning: $REMAINING processes may still be running"
    echo "   Check Task Manager (Windows) or 'ps aux | grep python' (Unix/Linux)"
else
    echo "✅ All target processes appear to be cleaned"
fi

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

echo "🔧 Starting RQ Workers..."
COMMON_LOG="logs/workers.log"
rm -f "$COMMON_LOG" 2>/dev/null || true

echo "🔍 Setting up separation workers..."
SEPARATION_WORKERS=${MAX_SEPARATION_WORKERS:-1}

echo "Starting ${SEPARATION_WORKERS} separation worker(s)..."
for i in $(seq 1 $SEPARATION_WORKERS); do
    echo "  - Starting sep_worker_${i}..."
    nohup python workers_starter.py separation_queue sep_worker_${i} redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &
    sleep 1
done

echo "🔍 Setting up dub orchestration workers..."
DUB_WORKERS=${MAX_DUB_ORCHESTRATION_WORKERS:-1}

echo "Starting ${DUB_WORKERS} dub orchestration worker(s)..."
for i in $(seq 1 $DUB_WORKERS); do
    echo "  - Starting dub_worker_${i}..."
    nohup python workers_starter.py dub_queue dub_worker_${i} redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &
    sleep 1
done

echo "Starting billing worker..."
nohup python workers_starter.py billing_queue billing_worker_1 redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &

echo "🎬 Starting video processing workers..."
VIDEO_WORKERS=${MAX_VIDEO_PROCESSING_WORKERS:-1}
echo "Starting ${VIDEO_WORKERS} video processing worker(s)..."
for i in $(seq 1 $VIDEO_WORKERS); do
    echo "  - Starting video_worker_${i}..."
    nohup python workers_starter.py video_processing_queue video_worker_${i} redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &
    sleep 1
done

echo "🎯 Starting VRAM service workers..."

echo "  - Starting WhisperX service workers (2 parallel VRAM workers)..."
# nohup python workers_starter.py whisperx_service_queue whisperx_service_worker_1 redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &
# LOAD_WHISPERX_MODEL=true LOAD_FISH_SPEECH_MODEL=false nohup python workers_starter.py whisperx_service_queue whisperx_service_worker_2 redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &

echo "  - Starting Fish Speech service worker (VRAM serial)..." # manully off for debugging
LOAD_WHISPERX_MODEL=false LOAD_FISH_SPEECH_MODEL=true nohup python workers_starter.py fish_speech_service_queue fish_speech_service_worker_1 redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &

echo "⏳ Waiting for workers to initialize..."
sleep 8

echo "📊 Checking worker status..."
python check_workers.py

echo ""
echo "⏳ Allowing additional startup time for VRAM workers..."
sleep 5

echo "📊 Final worker status verification..."
python check_workers.py

echo "📋 Checking worker logs for errors..."
if [ -f "logs/workers.log" ]; then
    echo "Last 15 lines of worker log:"
    tail -15 logs/workers.log
    echo ""
    echo "🔍 Checking for any worker startup errors..."
    if grep -q "ERROR\|Failed\|Exception" logs/workers.log; then
        echo "⚠️ Errors detected in worker logs - check logs/workers.log for details"
    else
        echo "✅ No errors detected in worker logs"
    fi
else
    echo "❌ No worker log file found"
fi

echo ""
echo "📈 Worker Summary:"
TOTAL_EXPECTED=$((1 + DUB_WORKERS + 1 + VIDEO_WORKERS + 1 + 1))  # sep + dub + billing + video + whisperx + fish
echo "  - Expected workers: ${TOTAL_EXPECTED} (1 sep + ${DUB_WORKERS} dub + 1 billing + ${VIDEO_WORKERS} video + 1 whisperx + 1 fish)"
echo "  - Separation: 1 worker"
echo "  - Dub orchestration: ${DUB_WORKERS} workers (no AI models)"
echo "  - Billing: 1 worker"
echo "  - Video processing: ${VIDEO_WORKERS} workers"
echo "  - WhisperX service: 2 workers (parallel VRAM optimized)"
echo "  - Fish Speech service: 1 worker (VRAM serial)"

echo ""
echo "🎉 WSL Ubuntu Worker Setup Complete!"
echo ""
echo "📊 Monitor Commands:"
echo "   tail -f logs/workers.log         # Monitor all workers"
echo "   python check_workers.py          # Check worker status"
echo "   tail -f logs/info.log            # Monitor API server"
echo ""
echo "🔴 Stop Commands (Cross-Platform):"
echo "   # Windows:"
echo "   taskkill /F /IM python.exe       # Force stop all Python processes"
echo "   wmic process where \"name='python.exe'\" delete  # Alternative Windows method"
echo ""
echo "   # Unix/Linux/WSL:"
echo "   pkill -f 'python.*main.py'      # Stop API server"
echo "   pkill -f 'rq.*worker'           # Stop all workers"
echo "   pkill -f 'workers_starter.py'   # Stop worker starters"
echo "   pkill -f 'video_processing_worker' # Stop video workers"
echo ""
echo "🔧 WSL Redis Commands:"
echo "   sudo service redis-server start    # Start Redis"
echo "   sudo service redis-server stop     # Stop Redis"
echo "   sudo service redis-server status   # Check Redis"
echo ""
echo "🚀 Starting API server..."
mkdir -p logs

# Start main API with better logging
python main.py 
API_PID=$!

echo "✅ API server started with PID: $API_PID"
echo ""
echo "⏳ Initializing API server..."
sleep 5

if pgrep -f "python.*main.py" > /dev/null; then
    echo "✅ API server is running"
    echo ""
    echo "🌐 Service URLs:"
    echo "   API: http://localhost:8000"
    echo "   Docs: http://localhost:8000/docs"
    echo "   Health: http://localhost:8000/health"
else
    echo "⚠️ API server startup may have failed, check logs/info.log"
fi

echo ""
echo "✅ All services started successfully!"
echo "📋 Use 'tail -f logs/info.log' to monitor the API server"
echo "📋 Use 'tail -f logs/workers.log' to monitor workers"