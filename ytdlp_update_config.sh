#!/bin/bash
# Configuration for yt-dlp auto-update script
# Edit this file to match your server setup

# Restart command - Choose ONE option based on your setup:

# Option 1: systemctl (Ubuntu/Debian with systemd service)
# RESTART_CMD="sudo systemctl restart runpod-backend"

# Option 2: PM2 (if you're using PM2 process manager)
# RESTART_CMD="pm2 restart runpod-backend"

# Option 3: Supervisor (if you're using supervisord)
# RESTART_CMD="supervisorctl restart runpod-backend"

# Option 4: Docker restart (if running in Docker)
# RESTART_CMD="docker restart runpod-backend"

# Option 5: Basic kill and restart (default)
RESTART_CMD="pkill -f 'uvicorn main:app' && sleep 2 && cd '$SCRIPT_DIR' && nohup python main.py > logs/server.log 2>&1 &"

# Option 6: No restart (only update, manual restart needed)
# RESTART_CMD=""

# Note: Script checks DAILY but only updates when new version is available
# Crontab schedule: 0 3 * * * (every day at 3 AM)
# Only restarts server when an actual update happens

