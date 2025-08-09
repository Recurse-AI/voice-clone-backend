import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler

# Create logs directory
logs_dir = Path(__file__).resolve().parent.parent / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)

# Log Rotation Configuration:
# - Max size: 5MB per file (5*1024*1024 bytes)
# - Backup count: 2 (total 3 files max)
# - Auto deletion: oldest files removed when limit exceeded

# Log file paths
info_log = logs_dir / "info.log"
error_log = logs_dir / "error.log"
debug_log = logs_dir / "debug.log"

# Create logger
logger = logging.getLogger("app_logger")
# Stream (console) handler
console_handler = logging.StreamHandler()

logger.setLevel(logging.DEBUG)  # Capture all levels

# Clear existing handlers to avoid duplication during reloads (dev server)
if logger.hasHandlers():
    logger.handlers.clear()

# Formatter
formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Info handler with 5MB rotation
info_handler = RotatingFileHandler(
    info_log, 
    maxBytes=5*1024*1024,  # 5MB
    backupCount=2,         # Keep 2 backup files
    encoding='utf-8'
)
info_handler.setLevel(logging.INFO)
info_handler.setFormatter(formatter)

# Debug handler with 5MB rotation
debug_handler = RotatingFileHandler(
    debug_log, 
    maxBytes=5*1024*1024,  # 5MB
    backupCount=2,         # Keep 2 backup files
    encoding='utf-8'
)
debug_handler.setLevel(logging.DEBUG)
debug_handler.setFormatter(formatter)

# Error handler with 5MB rotation
error_handler = RotatingFileHandler(
    error_log, 
    maxBytes=5*1024*1024,  # 5MB
    backupCount=2,         # Keep 2 backup files
    encoding='utf-8'
)
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(formatter)

# Add all handlers
logger.addHandler(info_handler)
logger.addHandler(debug_handler)
logger.addHandler(error_handler)
logger.addHandler(console_handler)