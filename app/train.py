import sys
sys.path.append('./app/ai-toolkit')
from toolkit.job import run_job
from train_config import train_config
import logging

# Set up logger


try:
    logging.info("Starting FLUX training job")
    logging.info(f"Training configuration: {train_config['config']['name']}")
    
    run_job(train_config)
    
    logging.info("Training completed successfully")
except Exception as e:
    logging.error(f"Error during training: {str(e)}")
    raise