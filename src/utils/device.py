from __future__ import annotations
import torch

def get_device(prefer: str = "cuda") -> torch.device:
    prefer = (prefer or "cuda").lower()
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
