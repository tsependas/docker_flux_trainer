import sys
sys.path.append('./app/ai-toolkit')
from toolkit.job import run_job
from train_config import train_config
from logger_config import setup_logger

# Set up logger
logger = setup_logger('flux_trainer')

try:
    logger.info("Starting FLUX training job")
    logger.info(f"Training configuration: {train_config['config']['name']}")
    
    run_job(train_config)
    
    logger.info("Training completed successfully")
except Exception as e:
    logger.error(f"Error during training: {str(e)}")
    raise