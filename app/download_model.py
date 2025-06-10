from huggingface_hub import snapshot_download
import os
import logging

# Set up logger
logger = logging.getLogger('model_downloader')

# Retrieve the Hugging Face token from the environment
hf_token = os.environ.get("HF_TOKEN")
if hf_token is None:
    logger.error("HF_TOKEN environment variable is not set.")
    raise ValueError("HF_TOKEN environment variable is not set.")

# Define model ID and save directory
model_id = "black-forest-labs/FLUX.1-dev"
save_dir = "./workspace/flux1-dev"

logger.info(f"Starting download of model {model_id} to {save_dir}")

try:
    # Step 1: Download (skip big files if needed)
    snapshot_download(
        repo_id=model_id,
        token=hf_token,
        local_dir=save_dir,
        cache_dir=save_dir,
        ignore_patterns=["flux1-dev.safetensors", "ae.safetensors"]
    )
    logger.info("Model download completed successfully")
except Exception as e:
    logger.error(f"Error during model download: {str(e)}")
    raise
