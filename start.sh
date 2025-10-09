#!/bin/bash

echo "🚀 Starting API + Workers (Debug Mode)..."

source venv/Scripts/activate 2>/dev/null || source venv/bin/activate

export LOAD_AI_MODELS=true
export DEBUG_MODE=true

#clear all workers
python cleanup_workers.py

mkdir -p logs tmp

#kill all workers
pkill -9 -f "python.*worker" 2>/dev/null || true
pkill -9 -f "rq.*worker" 2>/dev/null || true
pkill -9 -f "uvicorn" 2>/dev/null || true
redis-cli shutdown 2>/dev/null || true
pkill -9 -f "redis-server" 2>/dev/null || true
fuser -k 8000/tcp 6379/tcp 8080/tcp 5000/tcp 2>/dev/null || true


# echo "🚀 Starting API server..."
# python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload > logs/api_debug.log 2>&1 &
# API_PID=$!

# sleep 3

# echo "🔧 Starting workers..."
# python workers_starter.py dub_queue dub_worker_1 > logs/dub_worker.log 2>&1 &
# sleep 1
# python workers_starter.py billing_queue billing_worker_1 > logs/billing_worker.log 2>&1 &
# sleep 1
# python workers_starter.py whisperx_service_queue whisperx_service_worker_1 > logs/whisperx_service_worker.log 2>&1 &
# sleep 1

# echo ""
# echo "✅ Running!"
# echo ""
# echo "API: http://localhost:8000/docs"
# echo "Health: http://localhost:8000/health/ready"
# echo ""
# echo "Logs:"
# echo "  tail -f logs/api_debug.log"
# echo "  tail -f logs/dub_worker.log"
# echo ""
