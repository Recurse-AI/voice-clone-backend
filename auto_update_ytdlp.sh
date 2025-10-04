#!/bin/bash
# Auto-update yt-dlp script - Checks daily, updates only when new version available
# Usage: Add to crontab: 0 3 * * * /path/to/auto_update_ytdlp.sh (runs daily at 3 AM)

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
    
    # Verify update
    NEW_VERSION=$(python -c "import yt_dlp; print(yt_dlp.version.__version__)" 2>/dev/null)
    
    if [ "$CURRENT_VERSION" != "$NEW_VERSION" ]; then
        echo "[$(date)] ✓ yt-dlp updated: $CURRENT_VERSION → $NEW_VERSION" >> "$LOG_FILE"
        
        if [ -n "$RESTART_CMD" ]; then
            echo "[$(date)] Restarting server..." >> "$LOG_FILE"
            eval "$RESTART_CMD"
            echo "[$(date)] ✓ Server restarted successfully" >> "$LOG_FILE"
        else
            echo "[$(date)] ⚠ Update complete. Manual server restart required." >> "$LOG_FILE"
        fi
    else
        echo "[$(date)] ⚠ Update failed or already at latest version" >> "$LOG_FILE"
    fi
else
    # No update needed - keep log minimal (only log every 7 days to avoid spam)
    if [ ! -f "logs/.last_check" ] || [ $(find logs/.last_check -mtime +6 2>/dev/null | wc -l) -gt 0 ]; then
        echo "[$(date)] ✓ yt-dlp up to date ($CURRENT_VERSION)" >> "$LOG_FILE"
        touch logs/.last_check
    fi
fi

deactivate
