#!/bin/bash
# Auto-update yt-dlp script - Runs weekly to keep yt-dlp updated
# Usage: Add to crontab: 0 3 * * 0 /path/to/auto_update_ytdlp.sh

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

echo "[$(date)] Starting yt-dlp update check..." >> "$LOG_FILE"

# Activate virtual environment
source venv/bin/activate

# Get current version
CURRENT_VERSION=$(python -c "import yt_dlp; print(yt_dlp.version.__version__)" 2>/dev/null)
echo "[$(date)] Current version: $CURRENT_VERSION" >> "$LOG_FILE"

# Update yt-dlp
pip install --upgrade yt-dlp --quiet

# Get new version
NEW_VERSION=$(python -c "import yt_dlp; print(yt_dlp.version.__version__)" 2>/dev/null)
echo "[$(date)] New version: $NEW_VERSION" >> "$LOG_FILE"

# Check if version changed
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
    echo "[$(date)] ✓ yt-dlp already up to date ($CURRENT_VERSION)" >> "$LOG_FILE"
fi

deactivate
