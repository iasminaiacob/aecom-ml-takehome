from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from datasets import load_dataset, DatasetDict

@dataclass(frozen=True)
class Food101Data:
    ds: DatasetDict
    label2name: Dict[int, str]
    name2label: Dict[str, int]

def load_food101(hf_cache_dir: str | Path | None = None) -> Food101Data:
    """
    Loads Food-101 from Hugging Face datasets.
    - If hf_cache_dir is provided, it uses it.
    - Else: rely on HF env vars (HF_HOME, HF_DATASETS_CACHE, etc.) or HF defaults.
    """
    ds_kwargs = {}
    if hf_cache_dir:
        ds_kwargs["cache_dir"] = str(hf_cache_dir)

    ds = load_dataset("ethz/food101", **ds_kwargs)

    class_label = ds["train"].features["label"]
    names = list(class_label.names)
    label2name = {i: n for i, n in enumerate(names)}
    name2label = {n: i for i, n in enumerate(names)}

    return Food101Data(ds=ds, label2name=label2name, name2label=name2label)