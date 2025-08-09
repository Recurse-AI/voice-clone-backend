import os
import logging
from logging.config import dictConfig
from logging.handlers import RotatingFileHandler

def setup_logging():
    """
    Setup logging with file rotation to prevent large log files.
    
    Configuration:
    - Maximum file size: 5MB per log file
    - Backup count: 2 (keeps total 3 files: current + 2 backups)
    - Files: info.log, debug.log, error.log
    - When limit reached: info.log → info.log.1 → info.log.2 → deleted
    """
    os.makedirs("app/logs", exist_ok=True)

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "customFormatter": {
                "format": "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "customFormatter",
                "level": "DEBUG",
            },
            "info_file_handler": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": "app/logs/info.log",
                "formatter": "customFormatter",
                "level": "INFO",
                "maxBytes": 5242880,  # 5MB = 5 * 1024 * 1024 bytes
                "backupCount": 2,     # Keep 2 backup files, total 3 files max
                "encoding": "utf-8"
            },
            "debug_file_handler": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": "app/logs/debug.log",
                "formatter": "customFormatter",
                "level": "DEBUG",
                "maxBytes": 5242880,  # 5MB
                "backupCount": 2,     # Keep 2 backup files
                "encoding": "utf-8"
            },
            "error_file_handler": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": "app/logs/error.log",
                "formatter": "customFormatter",
                "level": "ERROR",
                "maxBytes": 5242880,  # 5MB
                "backupCount": 2,     # Keep 2 backup files
                "encoding": "utf-8"
            },
        },
        "loggers": {
            "motor": {
                "level": "WARNING",
                "handlers": ["console"],
                "propagate": False
            },
            "pymongo": {
                "level": "WARNING",
                "handlers": ["console"],
                "propagate": False
            },
            "": {
                "level": "DEBUG",
                "handlers": [
                    "console",
                    "info_file_handler",
                    "debug_file_handler",
                    "error_file_handler"
                ],
            },
        },
    }

    dictConfig(logging_config)
