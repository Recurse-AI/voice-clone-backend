import os
import logging
import warnings
from logging.config import dictConfig

def setup_logging():
    from app.config.settings import settings
    os.makedirs("app/logs", exist_ok=True)
    
    # Suppress third-party noise
    warnings.filterwarnings("ignore")
    
    # Suppress verbose loggers
    for logger_name in ["urllib3", "requests", "huggingface_hub", "transformers", 
                       "speechbrain", "pyannote", "torchaudio", "matplotlib", "tqdm"]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    
    # Disable tqdm progress bars
    os.environ['TQDM_DISABLE'] = '1'

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
                "level": "INFO",
            },
            "info_file_handler": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": "app/logs/info.log",
                "formatter": "customFormatter",
                "level": "INFO",
                "maxBytes": 5242880,
                "backupCount": 2,
                "encoding": "utf-8"
            },

            "error_file_handler": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": "app/logs/error.log",
                "formatter": "customFormatter",
                "level": "ERROR",
                "maxBytes": 5242880,
                "backupCount": 2,
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
                "level": "INFO",
                "handlers": [
                    "console",
                    "info_file_handler",
                    "error_file_handler"
                ],
            },
        },
    }

    dictConfig(logging_config)
