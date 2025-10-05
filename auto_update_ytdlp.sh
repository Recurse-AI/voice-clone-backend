#!/bin/bash
# Auto-update yt-dlp script - Checks monthly, updates only when new version available
# Usage: Add to crontab: 0 3 1 * * /path/to/auto_update_ytdlp.sh (runs 1st of every month at 3 AM)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LOG_FILE="logs/ytdlp_updates.log"
mkdir -p logs

# Load configuration
if [ -f "ytdlp_update_config.sh" ]; then
    source ytdlp_update_config.sh
else
    # Default: basic restart
    RESTART_CMD="pkill -f 'uvicorn main:app' && sleep 2 && cd '$SCRIPT_DIR' && nohup python main.py > logs/server.log 2>&1 &"
fi

# Activate virtual environment
source venv/bin/activate

# Get current installed version
CURRENT_VERSION=$(python -c "import yt_dlp; print(yt_dlp.version.__version__)" 2>/dev/null)

# Check latest available version on PyPI without installing
LATEST_VERSION=$(pip index versions yt-dlp 2>/dev/null | grep -oP 'Available versions: \K[0-9.]+' | head -1)

# Fallback: if pip index doesn't work, check PyPI JSON API
if [ -z "$LATEST_VERSION" ]; then
    LATEST_VERSION=$(curl -s https://pypi.org/pypi/yt-dlp/json | python -c "import sys, json; print(json.load(sys.stdin)['info']['version'])" 2>/dev/null)
fi

echo "[$(date)] Version check: Current=$CURRENT_VERSION, Latest=$LATEST_VERSION" >> "$LOG_FILE"

# Compare versions
if [ "$CURRENT_VERSION" != "$LATEST_VERSION" ] && [ -n "$LATEST_VERSION" ]; then
    echo "[$(date)] 🔄 New version available! Updating..." >> "$LOG_FILE"
    
    # Update yt-dlp
    pip install --upgrade yt-dlp --quiet
    
    # Also update PO Token plugin
    echo "[$(date)] Updating PO Token provider plugin..." >> "$LOG_FILE"
    pip install --upgrade bgutil-ytdlp-pot-provider --quiet
    
    # Verify update
    NEW_VERSION=$(python -c "import yt_dlp; print(yt_dlp.version.__version__)" 2>/dev/null)
    
    if [ "$CURRENT_VERSION" != "$NEW_VERSION" ]; then
        echo "[$(date)] ✓ yt-dlp and PO Token plugin updated: $CURRENT_VERSION → $NEW_VERSION" >> "$LOG_FILE"
        
        echo "[$(date)] Restarting server..." >> "$LOG_FILE"
        pkill -f "uvicorn.*main:app" 2>/dev/null || true
        sleep 3
        
        cd "$(dirname "$0")"
        nohup ./venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4 > logs/info.log 2>&1 &
        
        echo "[$(date)] ✓ Server restarted successfully" >> "$LOG_FILE"
    else
        echo "[$(date)] ⚠ Update failed or already at latest version" >> "$LOG_FILE"
    fi
else
    echo "[$(date)] ✓ yt-dlp up to date ($CURRENT_VERSION)" >> "$LOG_FILE"
fi

deactivate
