#!/bin/bash

# Clean API Startup Script

echo "Installing dependencies..."
#activate venv
pip install rq redis
# pip install -r requirements.txt || trueve


echo "Setting up Redis..."
if ! command -v redis-server &> /dev/null; then
    echo "Redis not found - using external Redis"
fi

echo "Stopping existing processes..."
# Kill uvicorn/main and any lingering rq workers
pkill -f "python.*main.py" 2>/dev/null || true
pkill -f "rq.*worker" 2>/dev/null || true
pkill -f "workers_starter.py" 2>/dev/null || true

echo "Force cleaning all workers..."
python cleanup_workers.py
sleep 2

echo "Starting Redis server..."
if command -v redis-server &> /dev/null; then
    nohup redis-server > logs/redis.log 2>&1 &
    echo "Redis started"
else
    echo "Using external Redis"
fi

sleep 3
mkdir -p logs

echo "Starting RQ Workers..."

# Start separation worker
echo "Starting workers (using common log)..."
# All workers use same log file for easier monitoring
COMMON_LOG="logs/workers.log"
rm -f "$COMMON_LOG" 2>/dev/null || true

echo "Starting separation worker..."
nohup python workers_starter.py separation_queue sep_worker_1 redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &

echo "Starting dub workers..."
nohup python workers_starter.py dub_queue dub_worker_1 redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &
nohup python workers_starter.py dub_queue dub_worker_2 redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &

echo "Starting billing worker..."
nohup python workers_starter.py billing_queue billing_worker_1 redis://127.0.0.1:6379 >> "$COMMON_LOG" 2>&1 &

echo "Waiting for workers to initialize..."
sleep 5

echo "Checking worker status..."
python check_workers.py

echo "Starting API server..."
echo "Monitor workers: tail -f logs/workers.log"
echo "Stop all: pkill -f 'python.*main.py' && pkill -f 'rq.*worker'"
echo

# Start main API
python main.py

echo "Services started successfully!"