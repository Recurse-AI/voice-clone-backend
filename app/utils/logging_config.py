import os
import logging
from logging.config import dictConfig

def setup_logging():
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
                "class": "logging.FileHandler",
                "filename": "app/logs/info.log",
                "formatter": "customFormatter",
                "level": "INFO",
            },
            "debug_file_handler": {
                "class": "logging.FileHandler",
                "filename": "app/logs/debug.log",
                "formatter": "customFormatter",
                "level": "DEBUG",
            },
            "error_file_handler": {
                "class": "logging.FileHandler",
                "filename": "app/logs/error.log",
                "formatter": "customFormatter",
                "level": "ERROR",
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
