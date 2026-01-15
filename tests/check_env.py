import os

KEYS = [
    "HF_HOME",
    "HF_DATASETS_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
]

for k in KEYS:
    print(f"{k}={os.environ.get(k)}")