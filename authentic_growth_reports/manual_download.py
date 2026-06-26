from huggingface_hub import snapshot_download
import os

repo_id = "unsloth/gemma-4-e4b-it-unsloth-bnb-4bit"
local_dir = r"D:\Code\Grad\authentic_growth_reports\gemma_4_weights"

print(f"Starting manual download of {repo_id} to {local_dir}...")
try:
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.gguf"], # Don't need GGUF
    )
    print("Download complete!")
except Exception as e:
    print(f"Download failed: {e}")
