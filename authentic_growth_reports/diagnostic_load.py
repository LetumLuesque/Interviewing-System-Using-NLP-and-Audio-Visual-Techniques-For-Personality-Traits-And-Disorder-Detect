import os
os.environ["PYTHONUTF8"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
import torch
from unsloth import FastVisionModel

print("Loading model for diagnostic check...")
try:
    model, processor = FastVisionModel.from_pretrained(
        model_name = "unsloth/gemma-4-E4B-it",
        load_in_4bit = True,
    )
    print("Model loaded successfully!")
    print(f"Architecture: {type(model)}")
    print(f"Processor: {type(processor)}")
except Exception as e:
    print(f"Error loading model: {e}")
