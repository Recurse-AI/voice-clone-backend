#!/bin/bash

# Platform independent startup script for Voice Cloning API
# Works on Linux (RunPod) and Windows (Git Bash/WSL)

echo "🚀 Voice Cloning API Startup - Platform Independent"

# Set UTF-8 environment variables for Unicode support (Linux/RunPod)
export PYTHONIOENCODING=utf-8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# Detect platform more accurately
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    PLATFORM="windows"
    echo "📱 Detected: Windows (Git Bash/Cygwin)"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLATFORM="linux"
    echo "🐧 Detected: Linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="macos"
    echo "🍎 Detected: macOS"
else
    PLATFORM="unknown"
    echo "❓ Unknown platform: $OSTYPE"
fi

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found. Please create one based on .env.example"
    echo "   Or run: ./runpod_setup.sh"
    exit 1
fi

# Create temp directory based on platform
if [[ "$PLATFORM" == "windows" ]]; then
    mkdir -p ./tmp/voice_cloning
    export TEMP_DIR="./tmp/voice_cloning"
    echo "📁 Created temporary directory: ./tmp/voice_cloning"
else
    mkdir -p /tmp/voice_cloning
    export TEMP_DIR="/tmp/voice_cloning"
    echo "📁 Created temporary directory: /tmp/voice_cloning"
fi

# Create logs directory
mkdir -p ./logs

# Log rotation function
rotate_api_log() {
    local log_file="./logs/api.log"
    local max_lines=100
    
    if [ -f "$log_file" ]; then
        local current_lines=$(wc -l < "$log_file" 2>/dev/null || echo "0")
        
        if [ "$current_lines" -gt "$max_lines" ]; then
            echo "📋 Rotating log file (current: $current_lines lines, keeping last $max_lines)"
            
            # Keep only the last 100 lines
            tail -n "$max_lines" "$log_file" > "$log_file.tmp" && mv "$log_file.tmp" "$log_file"
            
            echo "✅ Log rotated successfully"
        else
            echo "📋 Log file size OK ($current_lines lines)"
        fi
    fi
}

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python -m venv venv
    
    # Activate virtual environment based on platform
    if [[ "$PLATFORM" == "windows" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
    
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
else
    echo "🔧 Activating virtual environment..."
    if [[ "$PLATFORM" == "windows" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
    
    echo "📦 Updating dependencies..."
    pip install -r requirements.txt
fi

# Check GPU availability
echo "🖥️ Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU detected, enabling CUDA"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
    export CUDA_AVAILABLE=true
else
    echo "⚠️ No GPU detected, using CPU (may be slow)"
    export CUDA_AVAILABLE=false
fi

# Test FFmpeg installation
echo "🎬 Testing FFmpeg installation..."
if command -v ffmpeg &> /dev/null; then
    ffmpeg -version | head -1
    echo "✅ FFmpeg is working"
elif [[ "$PLATFORM" == "windows" ]] && [ -f "./ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe" ]; then
    ./ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe -version | head -1
    echo "✅ FFmpeg is working (local installation)"
    export PATH="$PWD/ffmpeg-master-latest-win64-gpl/bin:$PATH"
else
    echo "❌ FFmpeg not found! Please install FFmpeg first."
    if [[ "$PLATFORM" == "windows" ]]; then
        echo "   Windows: Download from https://ffmpeg.org/download.html"
        echo "   Or run: ./runpod_setup.sh"
        echo "   Or use: choco install ffmpeg"
    else
        echo "   Linux: sudo apt-get install ffmpeg"
        echo "   Or run: ./runpod_setup.sh"
    fi
    exit 1
fi

# Check required environment variables
echo "🔍 Checking environment variables..."
source .env 2>/dev/null || true

if [ -z "$R2_ACCESS_KEY_ID" ] || [ -z "$R2_SECRET_ACCESS_KEY" ] || [ -z "$R2_BUCKET_NAME" ]; then
    echo "⚠️ Warning: R2 Storage credentials not set. Video processing may fail."
fi

if [ -z "$ASSEMBLYAI_API_KEY" ]; then
    echo "⚠️ Warning: AssemblyAI API key not set. Voice cloning may fail."
fi

if [ -z "$API_ACCESS_TOKEN" ] || [ -z "$RUNPOD_ENDPOINT_ID" ]; then
    echo "⚠️ Warning: RunPod credentials not set. Audio separation may fail."
fi

# Rotate logs if needed
echo "📋 Checking log file size..."
rotate_api_log

# Kill any existing processes
echo "🔄 Stopping any existing processes..."
if [[ "$PLATFORM" == "windows" ]]; then
    taskkill /F /IM python.exe 2>/dev/null || true
    taskkill /F /IM uvicorn.exe 2>/dev/null || true
else
    pkill -f "python main.py" || true
    pkill -f "uvicorn" || true
fi
sleep 2

# Set environment variables for the current session
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Start the API
echo "🎵 Starting Voice Cloning API on port 8000..."
echo "🌐 API will be available at: http://localhost:8000"
echo "📖 Press Ctrl+C to stop the server"
echo "📋 Logs will appear below:"
echo "----------------------------------------"

python main.py 