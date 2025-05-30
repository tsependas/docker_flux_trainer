import os
from huggingface_hub import snapshot_download

HF_TOKEN = os.environ.get("HF_TOKEN")

print("📥 Downloading $MODEL_ID to '$SAVE_DIR' ...")
snapshot_download(
    repo_id="black-forest-labs/FLUX.1-dev",
    token=HF_TOKEN,
    local_dir="/app/flux1-dev",
    local_dir_use_symlinks=False,
    ignore_patterns=[
        "flux1-dev.safetensors",
        "ae.safetensors"
    ]
)
print("✅ Download complete.")