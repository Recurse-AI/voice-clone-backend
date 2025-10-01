#!/bin/bash

echo "🔧 DEBUG MODE - Clip Generation API"

echo "🧹 Cleanup existing processes..."
pkill -f "python.*main.py" 2>/dev/null || true
pkill -f "uvicorn.*main:app" 2>/dev/null || true
pkill -f "rq.*worker" 2>/dev/null || true
pkill -f "workers_starter.py" 2>/dev/null || true
sleep 2
mkdir -p logs

echo "🔧 Starting Workers..."
CLIP_LOG="logs/clip_worker.log"
BILLING_LOG="logs/billing_worker.log"
rm -f "$CLIP_LOG" "$BILLING_LOG" 2>/dev/null || true

# Start Clip Worker
echo "  Starting clip_worker_1..."
nohup python workers_starter.py clip_queue clip_worker_1 redis://127.0.0.1:6379 >> "$CLIP_LOG" 2>&1 &
CLIP_PID=$!
sleep 2

# Start Billing Worker
echo "  Starting billing_worker_1..."
nohup python workers_starter.py billing_queue billing_worker_1 redis://127.0.0.1:6379 >> "$BILLING_LOG" 2>&1 &
BILLING_PID=$!
sleep 2

echo "  Clip Worker PID: $CLIP_PID"
echo "  Billing Worker PID: $BILLING_PID"

echo "📊 Worker Status:"
python check_workers.py 2>/dev/null || echo "  Workers started (clip + billing)"

echo ""
echo "🚀 Starting API Server (DEBUG MODE with AUTO-RELOAD)..."
echo "📋 API Log: logs/info.log"
echo "📋 Clip Worker Log: logs/clip_worker.log"
echo "📋 Billing Worker Log: logs/billing_worker.log"
echo ""
echo "🌐 URLs:"
echo "   API: http://localhost:8000"
echo "   Docs: http://localhost:8000/docs"
echo ""

#run with reload and story log in logs/info.log
uvicorn main:app --host 0.0.0.0 --port 8000 --reload --log-level debug > logs/info.log 2>&1
