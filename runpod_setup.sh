#!/bin/bash

rm -rf logs/* tmp/*

echo "🚀 Starting Voice Cloning API on RunPod..."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FFMPEG_USE_GPU=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_HOME=/usr
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/bin:/usr/local/cuda/bin:$PATH
export FORCE_CUDA=1

echo "🐍 Activating virtual environment..."
source venv/bin/activate

echo "🔧 Upgrading yt-dlp..."
pip install --upgrade yt-dlp --quiet
pip install --upgrade bgutil-ytdlp-pot-provider --quiet

echo "⏰ Setting up auto-update schedule (3 AM on 1st of every month)..."
chmod +x auto_update_ytdlp.sh
SCRIPT_PATH="$(pwd)/auto_update_ytdlp.sh"
(crontab -l 2>/dev/null | grep -v "auto_update_ytdlp.sh"; echo "0 3 1 * * $SCRIPT_PATH") | crontab - 2>/dev/null || true

mkdir -p ./tmp ./logs
chmod 755 ./tmp ./logs

echo "🧹 Cleanup previous processes..."
pkill -9 -f "python.*main" 2>/dev/null || true
pkill -9 -f "python.*worker" 2>/dev/null || true
pkill -9 -f "rq.*worker" 2>/dev/null || true
pkill -9 -f "uvicorn" 2>/dev/null || true
redis-cli shutdown 2>/dev/null || true
pkill -9 -f "redis-server" 2>/dev/null || true

fuser -k 8000/tcp 6379/tcp 8080/tcp 5000/tcp 2>/dev/null || true

rm -rf ./tmp/* ./logs/*.pid ./logs/*.log 2>/dev/null || true
rm -f dump.rdb appendonly.aof *.rdb *.aof 2>/dev/null || true

if command -v nvidia-smi >/dev/null 2>&1; then
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
fi

sleep 1

echo "🔐 Checking PO Token Plugin..."
if python -c "import bgutil_ytdlp_pot_provider" 2>/dev/null; then
    echo "✅ PO Token Plugin installed"
else
    pip install --upgrade bgutil-ytdlp-pot-provider --quiet
fi

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

echo "🚀 Starting API server..."
WORKERS=${WORKERS:-4}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}

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

WHISPER_DEVICE=cpu WHISPER_COMPUTE_TYPE=float32 nohup ./venv/bin/python workers_starter.py cpu_whisperx_service_queue cpu_whisperx_1 redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &
FISH_SPEECH_DEVICE=cpu FISH_SPEECH_PRECISION=float32 nohup ./venv/bin/python workers_starter.py cpu_fish_speech_service_queue cpu_fish_speech_1 redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &

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