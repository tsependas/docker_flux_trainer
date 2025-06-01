#import torch
#torch.backends.cudnn.benchmark = True
#torch.backends.cuda.matmul.allow_tf32 = True
#torch.backends.cudnn.allow_tf32 = True


import sys
sys.path.append('/app/ai-toolkit')
from toolkit.job import run_job
from train_config import train_config


run_job(train_config)