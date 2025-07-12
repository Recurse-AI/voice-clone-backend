#!/bin/bash

# RunPod Voice Cloning API Setup - Production Ready

echo "🚀 Setting up Voice Cloning API on RunPod GPU..."

# Fix /tmp directory permissions first
echo "🔧 Fixing system permissions..."
chmod 1777 /tmp 2>/dev/null || true
mkdir -p /tmp && chmod 1777 /tmp 2>/dev/null || true
export TMPDIR=/tmp

# Set working directory to the project directory
if [ -f "requirements.txt" ]; then
    echo "📁 Found requirements.txt in current directory"
elif [ -f "voice-clone-backend/requirements.txt" ]; then
    echo "📁 Changing to voice-clone-backend directory"
    cd voice-clone-backend
elif [ -f "/workspace/voice-clone-backend/requirements.txt" ]; then
    echo "📁 Changing to /workspace/voice-clone-backend directory"
    cd /workspace/voice-clone-backend
else
    echo "❌ Could not find requirements.txt file"
    echo "Current directory: $(pwd)"
    echo "Files in current directory:"
    ls -la
    exit 1
fi

echo "📁 Working directory: $(pwd)"

# Install system dependencies with better error handling
echo "📦 Installing system dependencies..."
export DEBIAN_FRONTEND=noninteractive

# Try to fix APT issues first
echo "🔧 Fixing APT configuration..."
apt-get clean || true
rm -rf /var/lib/apt/lists/* || true
mkdir -p /var/lib/apt/lists/partial || true

# Try updating packages with retry logic
echo "🔄 Updating package lists..."
for i in {1..3}; do
    if apt-get update -y; then
        echo "✅ Package lists updated successfully"
        break
    else
        echo "⚠️  Attempt $i failed, retrying..."
        sleep 2
        apt-get clean || true
        rm -rf /var/lib/apt/lists/* || true
    fi
    
    if [ $i -eq 3 ]; then
        echo "⚠️  Package update failed, but continuing with existing packages..."
    fi
done

# Install dependencies with fallback
echo "📦 Installing required packages..."
apt-get install -y ffmpeg libsndfile1 python3-dev python3-pip python3-venv git curl build-essential || {
    echo "⚠️  Some packages failed to install, checking what's available..."
    
    # Try installing packages individually
    for package in ffmpeg libsndfile1 python3-dev python3-pip python3-venv git curl build-essential; do
        if apt-get install -y "$package"; then
            echo "✅ Installed $package"
        else
            echo "⚠️  Failed to install $package"
        fi
    done
}

apt-get autoremove -y || true

# Verify GPU availability
echo "🔍 Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
else
    echo "⚠️  GPU not detected, running on CPU"
fi

# Create directories with proper permissions
echo "📁 Creating directories..."
mkdir -p ./tmp/voice_cloning ./logs
chmod 755 ./tmp/voice_cloning ./logs

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
    else
        echo "📋 No existing log file found"
    fi
}

# Setup log rotation as background task
setup_log_rotation() {
    echo "⚙️  Setting up automatic log rotation..."
    
    # Create log rotation script
    cat > ./logs/rotate_logs.sh << 'EOF'
#!/bin/bash
# Auto log rotation script

LOG_FILE="./logs/api.log"
MAX_LINES=100
CHECK_INTERVAL=300  # 5 minutes

while true; do
    if [ -f "$LOG_FILE" ]; then
        CURRENT_LINES=$(wc -l < "$LOG_FILE" 2>/dev/null || echo "0")
        
        if [ "$CURRENT_LINES" -gt "$MAX_LINES" ]; then
            echo "$(date): Rotating log file ($CURRENT_LINES lines -> $MAX_LINES lines)" >> "./logs/rotation.log"
            tail -n "$MAX_LINES" "$LOG_FILE" > "$LOG_FILE.tmp" && mv "$LOG_FILE.tmp" "$LOG_FILE"
        fi
    fi
    
    sleep $CHECK_INTERVAL
done
EOF
    
    chmod +x ./logs/rotate_logs.sh
    
    # Kill any existing log rotation process
    pkill -f "rotate_logs.sh" || true
    
    # Start log rotation in background
    nohup ./logs/rotate_logs.sh > /dev/null 2>&1 &
    
    echo "✅ Automatic log rotation enabled (checks every 5 minutes)"
}

# Setup Python environment
echo "🐍 Setting up Python environment..."
python3 -m venv venv || { echo "❌ Failed to create virtual environment"; exit 1; }
source venv/bin/activate || { echo "❌ Failed to activate virtual environment"; exit 1; }
pip install --upgrade pip || { echo "❌ Failed to upgrade pip"; exit 1; }

# Install Python dependencies
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt || { echo "❌ Failed to install requirements"; exit 1; }

# Verify critical packages
echo "🔍 Verifying critical packages..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" || { echo "❌ PyTorch verification failed"; exit 1; }

# Create .env file
echo "⚙️  Creating configuration..."
cat > .env << EOF
# API Configuration
HOST=0.0.0.0
PORT=8000
CUDA_AVAILABLE=true
TEMP_DIR=./tmp/voice_cloning
LOGS_DIR=./logs

# R2 Storage
R2_ACCESS_KEY_ID=${R2_ACCESS_KEY_ID}
R2_SECRET_ACCESS_KEY=${R2_SECRET_ACCESS_KEY}
R2_BUCKET_NAME=${R2_BUCKET_NAME}
R2_ENDPOINT=${R2_ENDPOINT}
R2_PUBLIC_URL=${R2_PUBLIC_URL}

# API Keys
OPENAI_API_KEY=${OPENAI_API_KEY}
ASSEMBLYAI_API_KEY=${ASSEMBLYAI_API_KEY}
API_ACCESS_TOKEN=${API_ACCESS_TOKEN}
RUNPOD_ENDPOINT_ID=${RUNPOD_ENDPOINT_ID}
RUNPOD_TIMEOUT=${RUNPOD_TIMEOUT:-1800000}

# MongoDB
MONGODB_URI=${MONGODB_URI}

# Processing Options
ENABLE_SUBTITLES=true
ENABLE_INSTRUMENTS=true
EOF

# Kill existing processes
echo "🔄 Stopping existing processes..."
pkill -f "python.*main.py" || true
sleep 3

# Rotate logs before starting
echo "📋 Checking and rotating logs..."
rotate_api_log

# Setup automatic log rotation
setup_log_rotation

# Start API with proper virtual environment
echo "🚀 Starting API server..."
source venv/bin/activate
nohup ./venv/bin/python main.py > ./logs/api.log 2>&1 &

# Wait for API to start
echo "⏳ Waiting for API to start..."
sleep 10

echo "🎉 Setup complete! Your Voice Cloning API is ready!" 