from huggingface_hub import snapshot_download
#from diffusers import FluxPipeline
#import torch
import os

# Retrieve the Hugging Face token from the environment
hf_token = os.environ.get("HF_TOKEN")
if hf_token is None:
    raise ValueError("HF_TOKEN environment variable is not set.")

# Define model ID and save directory
model_id = "black-forest-labs/FLUX.1-dev"
save_dir = "/workspace/flux1-dev"




# Step 1: Download (skip big files if needed)
snapshot_download(
    repo_id=model_id,
    token=hf_token,
    local_dir=save_dir,
    ignore_patterns=["flux1-dev.safetensors", "ae.safetensors"]
)
