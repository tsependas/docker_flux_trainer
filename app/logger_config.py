import logging
import sys
import os
from datetime import datetime

def setup_logger(name, log_dir="logs"):
    """
    Set up a logger with both file and console handlers
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Remove all handlers if already present (to prevent duplicate logs)
    if logger.hasHandlers():
        logger.handlers.clear()

    # File formatter
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f'{name}_{timestamp}.log'), mode='a'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))

    # Console formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Optionally, force immediate flushing on StreamHandler (should be default, but for paranoia):
    console_handler.flush = sys.stdout.flush

    return logger
