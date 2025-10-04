#!/bin/bash

# Set working directory to script location
rm -rf logs/*
rm -rf tmp/*


# Simplified RunPod Startup Script
# Use environment_setup.sh first for initial setup

echo "🚀 Starting Voice Cloning API on RunPod..."

# Basic environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FFMPEG_USE_GPU=1
export CUDA_LAUNCH_BLOCKING=0
export TORCH_BACKENDS_CUDNN_DETERMINISTIC=0

# GPU Configuration - Force CUDA usage
export CUDA_DEVICE_ORDER=PCI_BUS_ID  
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"

# CUDA Runtime Environment - Use system-installed libraries
export CUDA_HOME=/usr
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/bin:/usr/local/cuda/bin:$PATH

# Force GPU detection 
export FORCE_CUDA=1

# AI Model optimization - Disabled compilation for faster processing
export FISH_SPEECH_COMPILE=true  # Enabled for better performance
export TORCH_JIT_LOG_LEVEL=ERROR
export TORCH_COMPILE_MODE=reduce-overhead
export TORCH_COMPILE_BACKEND=inductor



# Activate virtual environment
echo "🐍 Activating virtual environment..."
source venv/bin/activate

# Upgrade yt-dlp and PO Token provider
echo "🔧 Upgrading yt-dlp and PO Token provider..."
pip install --upgrade yt-dlp --quiet
pip install --upgrade bgutil-ytdlp-pot-provider --quiet

# Setup auto-update cron job for yt-dlp (runs daily at 3 AM)
echo "⏰ Setting up auto-update schedule..."
chmod +x auto_update_ytdlp.sh
SCRIPT_PATH="$(pwd)/auto_update_ytdlp.sh"
(crontab -l 2>/dev/null | grep -v "auto_update_ytdlp.sh"; echo "0 3 * * * $SCRIPT_PATH") | crontab - 2>/dev/null || true

# Create directories
mkdir -p ./tmp ./logs
chmod 755 ./tmp ./logs
# COMPREHENSIVE CLEANUP - Kill ALL previous processes
echo "🧹 Comprehensive cleanup of all previous processes..."

# Kill all Python processes related to this application
pkill -9 -f "python.*main" 2>/dev/null || true
pkill -9 -f "python.*worker" 2>/dev/null || true
pkill -9 -f "python.*workers_starter" 2>/dev/null || true
pkill -9 -f "python.*check_workers" 2>/dev/null || true
pkill -9 -f "python.*separation_worker" 2>/dev/null || true
pkill -9 -f "python.*dub_worker" 2>/dev/null || true
pkill -9 -f "python.*billing_worker" 2>/dev/null || true
pkill -9 -f "python.*clip_worker" 2>/dev/null || true
pkill -9 -f "python.*whisperx_service_worker" 2>/dev/null || true
pkill -9 -f "python.*fish_speech_service_worker" 2>/dev/null || true
pkill -9 -f "python.*video_processing_worker" 2>/dev/null || true
pkill -9 -f "python.*resume_worker" 2>/dev/null || true

# Kill all RQ workers (various patterns)
pkill -9 -f "rq.*worker" 2>/dev/null || true
pkill -9 -f "rqworker" 2>/dev/null || true
pkill -9 -f "rq worker" 2>/dev/null || true

# Kill all uvicorn processes (various patterns)
pkill -9 -f "uvicorn.*main:app" 2>/dev/null || true
pkill -9 -f "uvicorn" 2>/dev/null || true

# Kill all Redis processes
redis-cli shutdown 2>/dev/null || true
pkill -9 -f "redis-server" 2>/dev/null || true
pkill -9 -f "redis" 2>/dev/null || true

# Kill any remaining worker processes
pkill -9 -f "worker" 2>/dev/null || true
pkill -9 -f "starter" 2>/dev/null || true

# Kill BGUtil server if running
pkill -9 -f "bgutils-provider" 2>/dev/null || true
pkill -9 -f "node.*bgutils" 2>/dev/null || true

# Free all relevant ports
fuser -k 8000/tcp 2>/dev/null || true
fuser -k 6379/tcp 2>/dev/null || true
fuser -k 8080/tcp 2>/dev/null || true
fuser -k 5000/tcp 2>/dev/null || true
fuser -k 3000/tcp 2>/dev/null || true
fuser -k 4416/tcp 2>/dev/null || true

# Kill any processes using our ports (alternative method)
# Check if lsof is available, if not skip this step
if command -v lsof >/dev/null 2>&1; then
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    lsof -ti:6379 | xargs kill -9 2>/dev/null || true
    lsof -ti:8080 | xargs kill -9 2>/dev/null || true
    lsof -ti:5000 | xargs kill -9 2>/dev/null || true
    lsof -ti:3000 | xargs kill -9 2>/dev/null || true
    lsof -ti:4416 | xargs kill -9 2>/dev/null || true
fi

# Comprehensive file cleanup
echo "🧹 Cleaning up temporary files and logs..."
rm -rf ./tmp/* 2>/dev/null || true
rm -rf ./logs/*.pid 2>/dev/null || true
rm -rf ./logs/*.log 2>/dev/null || true
rm -f dump.rdb appendonly.aof 2>/dev/null || true
rm -f *.rdb *.aof 2>/dev/null || true

# Clear any hanging processes in background
jobs -p | xargs kill -9 2>/dev/null || true

# GPU memory cleanup if available
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "🧹 Clearing GPU memory..."
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
fi

# Final verification - ensure nothing is running
sleep 1
if pgrep -f "uvicorn\|python.*main\|redis-server\|rq.*worker" > /dev/null; then
    echo "⚠️ Some processes still running, forcing kill..."
    pgrep -f "uvicorn\|python.*main\|redis-server\|rq.*worker" | xargs kill -9 2>/dev/null || true
    sleep 2
fi

echo "✅ Comprehensive cleanup completed - all previous processes killed and cleaned"

# START BGUTIL SERVER FOR PO TOKEN
echo "🔐 Starting BGUtil PO Token Provider..."

# Check if Node.js is installed
if ! command -v node >/dev/null 2>&1; then
    echo "📦 Node.js not found, installing..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - || true
    apt-get install -y nodejs || true
    
    # Verify Node.js installation
    if command -v node >/dev/null 2>&1; then
        echo "✅ Node.js installed: $(node --version)"
    else
        echo "❌ Node.js installation failed"
        echo "⚠️ BGUtil server will not be available"
        echo "   YouTube downloads will use fallback system"
    fi
else
    echo "✅ Node.js already installed: $(node --version)"
fi

# Check if npm is available
if ! command -v npm >/dev/null 2>&1; then
    echo "❌ npm not found, cannot install bgutils-provider"
else
    echo "✅ npm available: $(npm --version)"
    
    # Check if bgutils-provider is installed
    if ! command -v bgutils-provider >/dev/null 2>&1; then
        echo "📦 Installing bgutils-provider..."
        
        # Try installation with visible errors
        if npm install -g bgutils-provider; then
            echo "✅ bgutils-provider installed successfully"
        else
            echo "❌ bgutils-provider installation failed"
            echo "   Trying alternative installation method..."
            
            # Try with --force flag
            npm install -g bgutils-provider --force || true
        fi
    else
        echo "✅ bgutils-provider already installed"
    fi
fi

# Start BGUtil server in background
if command -v bgutils-provider >/dev/null 2>&1; then
    echo "🚀 Starting BGUtil PO Token server on port 4416..."
    
    # Kill any existing bgutils-provider
    pkill -9 -f "bgutils-provider" 2>/dev/null || true
    sleep 1
    
    # Start in background
    nohup bgutils-provider > logs/bgutil.log 2>&1 &
    BGUTIL_PID=$!
    echo "   Started with PID: $BGUTIL_PID"
    
    # Wait for server to initialize
    sleep 3
    
    # Verify BGUtil server
    if curl -s http://127.0.0.1:4416/ping > /dev/null 2>&1; then
        echo "✅ BGUtil PO Token server running and responding"
        echo "   🎯 HD/4K YouTube formats now enabled!"
        echo "   📋 Log: logs/bgutil.log"
    else
        echo "⚠️ BGUtil server started but not responding"
        echo "   Check logs/bgutil.log for details"
        echo "   YouTube downloads will use fallback system"
    fi
else
    echo "⚠️ BGUtil server not available (bgutils-provider not found)"
    echo "   YouTube downloads will use fallback system (360p-720p)"
    echo ""
    echo "   To manually install:"
    echo "   1. npm install -g bgutils-provider"
    echo "   2. bgutils-provider &"
    echo "   3. curl http://127.0.0.1:4416/ping"
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
    nohup ./venv/bin/python workers_starter.py separation_queue sep_worker_${i} redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &
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
    nohup ./venv/bin/python workers_starter.py dub_queue dub_worker_${i} redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &
    sleep 1  # Quick stagger for clean startup
done

echo "  - Starting 2 dedicated RESUME workers for instant job resumption..."
nohup ./venv/bin/python workers_starter.py dub_queue resume_worker_1 redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &
sleep 1
nohup ./venv/bin/python workers_starter.py dub_queue resume_worker_2 redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &

echo "  - Starting billing worker..."
nohup ./venv/bin/python workers_starter.py billing_queue billing_worker_1 redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &

echo "🎬 Starting clip generation workers..."
CLIP_WORKERS=${MAX_CLIP_WORKERS:-2}
echo "  - Starting ${CLIP_WORKERS} clip worker(s)..."
for i in $(seq 1 $CLIP_WORKERS); do
    echo "    - Starting clip_worker_${i}..."
    nohup ./venv/bin/python workers_starter.py clip_queue clip_worker_${i} redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &
    sleep 1
done

# VRAM Service Workers (Serial Processing)
echo "🎯 Starting VRAM service workers..."

echo "  - Starting WhisperX service worker (1 worker for 16GB VRAM)..."
nohup ./venv/bin/python workers_starter.py whisperx_service_queue whisperx_service_worker_1 redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &

# echo "  - Starting WhisperX service worker (2nd worker for 16GB VRAM)..."
# LOAD_WHISPERX_MODEL=true LOAD_FISH_SPEECH_MODEL=false nohup ./venv/bin/python workers_starter.py whisperx_service_queue whisperx_service_worker_2 redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &

echo "  - Starting Fish Speech service worker (1 worker for 16GB VRAM)..."
nohup ./venv/bin/python workers_starter.py fish_speech_service_queue fish_speech_service_worker_1 redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &

# CPU Workers for Load Balancing (Simple)
echo "  - Starting CPU WhisperX worker..."
WHISPER_DEVICE=cpu WHISPER_COMPUTE_TYPE=float32 nohup ./venv/bin/python workers_starter.py cpu_whisperx_service_queue cpu_whisperx_1 redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &

echo "  - Starting CPU Fish Speech worker..."
FISH_SPEECH_DEVICE=cpu FISH_SPEECH_PRECISION=float32 nohup ./venv/bin/python workers_starter.py cpu_fish_speech_service_queue cpu_fish_speech_1 redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &

echo "🎬 Starting video processing workers..."
VIDEO_WORKERS=${MAX_VIDEO_PROCESSING_WORKERS:-2}
echo "  - Starting ${VIDEO_WORKERS} video processing worker(s)..."
for i in $(seq 1 $VIDEO_WORKERS); do
    echo "    - Starting video_worker_${i}..."
    nohup ./venv/bin/python workers_starter.py video_processing_queue video_worker_${i} redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &
    sleep 1
done

echo "⏳ Waiting for VRAM workers to load models..."
sleep 10

echo "⏳ Waiting for workers to initialize..."
sleep 5

echo "🔍 Verifying CUDA environment..."
python3 -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDNN Version: {torch.backends.cudnn.version()}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
else:
    print('⚠️ CUDA not available')
" || echo "CUDA verification completed"

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
echo "🔐 PO Token Server:"
if pgrep -f "bgutils-provider" > /dev/null 2>&1; then
    # Server is running, check if responding
    if curl -s http://127.0.0.1:4416/ping > /dev/null 2>&1; then
        echo "   ✅ BGUtil server: RUNNING & RESPONDING"
        echo "   🎯 HD/4K YouTube downloads: ENABLED"
        echo "   📊 Port: 4416"
    else
        echo "   ⚠️ BGUtil server: RUNNING but NOT RESPONDING"
        echo "   Check: tail -f logs/bgutil.log"
    fi
else
    echo "   ⚠️ BGUtil server: NOT RUNNING"
    
    # Check if bgutils-provider is installed
    if command -v bgutils-provider >/dev/null 2>&1; then
        echo "   📦 bgutils-provider: INSTALLED"
        echo "   🔧 To start: bgutils-provider &"
    else
        echo "   ❌ bgutils-provider: NOT INSTALLED"
        echo "   📦 To install: npm install -g bgutils-provider"
    fi
    
    echo "   ⚠️ YouTube downloads using fallback system (limited HD)"
fi
echo ""
echo "🔴 Stop Commands:"
echo "   pkill -f 'uvicorn.*main:app' && pkill -f 'rq.*worker'"
echo "   pkill -f 'video_processing_worker'  # Stop video workers"
echo "   pkill -f 'bgutils-provider'  # Stop PO Token server"
