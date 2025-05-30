import os
import torch
from diffusers import FluxPipeline

# Retrieve the Hugging Face token from the environment
hf_token = os.environ.get("HF_TOKEN")
if hf_token is None:
    raise ValueError("HF_TOKEN environment variable is not set.")

# Define model ID and save directory
model_id = "black-forest-labs/FLUX.1-dev"
save_dir = "/app/flux1-dev"

# Load the model with bfloat16 precision using the token
pipe = FluxPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    token=hf_token,
    resume_download=True,
    cache_dir=save_dir
)

# Save the model to the specified directory
pipe.save_pretrained(save_dir)