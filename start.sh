#!/bin/bash

echo "🚀 Starting API + Workers for ElevenLabs Dubbing Test..."

source venv/Scripts/activate 2>/dev/null || source venv/bin/activate

export LOAD_AI_MODELS=false
export DEBUG_MODE=true

# Clear all workers
python cleanup_workers.py

mkdir -p logs tmp

# Remove old log files
echo "🗑️  Cleaning up old logs and Redis files..."
rm -f logs/*.log 2>/dev/null || true
rm -f dump.rdb appendonly.aof *.rdb *.aof 2>/dev/null || true

# Kill all existing processes
pkill -9 -f "python.*worker" 2>/dev/null || true
pkill -9 -f "rq.*worker" 2>/dev/null || true
pkill -9 -f "uvicorn" 2>/dev/null || true
redis-cli shutdown 2>/dev/null || true
pkill -9 -f "redis-server" 2>/dev/null || true
fuser -k 8000/tcp 6379/tcp 8080/tcp 5000/tcp 2>/dev/null || true

sleep 2

# Start Redis (required for workers)
echo "🔴 Starting Redis..."
redis-server --daemonize yes --port 6379 --save "" --appendonly no
sleep 2

# Start API server
echo "🚀 Starting API server..."
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload > logs/api_debug.log 2>&1 &
API_PID=$!

sleep 3

# Start workers needed for ElevenLabs dubbing
echo "🔧 Starting workers..."

# 1. DUB WORKER - Main worker that processes dubbing (calls ElevenLabs service)
echo "   Starting dub_worker (main dubbing orchestrator)..."
python workers_starter.py dub_queue dub_worker_1 > logs/dub_worker.log 2>&1 &
sleep 1

# 2. BILLING WORKER - For credit management
echo "   Starting billing_worker (credit system)..."
python workers_starter.py billing_queue billing_worker_1 > logs/billing_worker.log 2>&1 &
sleep 1

echo ""
echo "✅ Running!"
echo ""
echo "📡 API: http://localhost:8000/docs"
echo "🏥 Health: http://localhost:8000/health/ready"
echo ""
echo "📋 Workers Running:"
echo "   - dub_worker (processes ElevenLabs dubbing)"
echo "   - billing_worker (manages credits)"
echo ""
echo "📝 Logs:"
echo "   tail -f logs/api_debug.log"
echo "   tail -f logs/dub_worker.log"
echo "   tail -f logs/billing_worker.log"
echo ""
echo "🧪 To test ElevenLabs dubbing:"
echo "   - Set model_type='excellent' in your request"
echo "   - Monitor: tail -f logs/dub_worker.log"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for Ctrl+C
wait