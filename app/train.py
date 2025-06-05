
import sys
sys.path.append('/app/ai-toolkit')
from toolkit.job import run_job
from train_config import train_config


run_job(train_config)