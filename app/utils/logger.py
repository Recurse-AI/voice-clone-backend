import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler

logs_dir = Path(__file__).resolve().parent.parent / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)

info_log = logs_dir / "info.log"
error_log = logs_dir / "error.log"

logger = logging.getLogger("app_logger")
console_handler = logging.StreamHandler()

logger.setLevel(logging.INFO)

if logger.hasHandlers():
    logger.handlers.clear()

formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

info_handler = RotatingFileHandler(
    info_log, 
    maxBytes=5*1024*1024,
    backupCount=2,
    encoding='utf-8'
)
info_handler.setLevel(logging.INFO)
info_handler.setFormatter(formatter)

# Debug handler removed to reduce log verbosity

error_handler = RotatingFileHandler(
    error_log, 
    maxBytes=5*1024*1024,
    backupCount=2,
    encoding='utf-8'
)
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(formatter)

logger.addHandler(info_handler)
logger.addHandler(error_handler)
logger.addHandler(console_handler)