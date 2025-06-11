import sys
sys.path.append('./ai-toolkit')
from toolkit.job import run_job
from train_config import train_config
import logging
from train_utils import get_task



try:
    logging.info("Starting FLUX training job")

    task = get_task()
    
    run_job(train_config)
    
    logging.info("Training completed successfully")
except Exception as e:
    logging.error(f"Error during training: {str(e)}")
    raise