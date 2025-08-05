import logging
from pathlib import Path

# Create logs directory
logs_dir = Path(__file__).resolve().parent.parent / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)

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

# Info handler
info_handler = logging.FileHandler(info_log)
info_handler.setLevel(logging.INFO)
info_handler.setFormatter(formatter)

# Debug handler
debug_handler = logging.FileHandler(debug_log)
debug_handler.setLevel(logging.DEBUG)
debug_handler.setFormatter(formatter)

# Error handler
error_handler = logging.FileHandler(error_log)
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(formatter)

# Add all handlers
logger.addHandler(info_handler)
logger.addHandler(debug_handler)
logger.addHandler(error_handler)
logger.addHandler(console_handler)