#!/bin/bash
# Auto-update yt-dlp script
# Runs daily via cron to keep yt-dlp updated

cd "$(dirname "$0")"
source venv/bin/activate
pip install --upgrade yt-dlp --quiet
echo "[$(date)] yt-dlp updated successfully" >> logs/ytdlp_updates.log

