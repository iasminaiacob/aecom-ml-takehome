from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
from torchvision.datasets import ImageFolder

def download_roboflow_classification_dataset(
    out_dir: str | Path,
    workspace: str,
    project: str,
    version: int,
    model_format: str = "folder",
    api_key_env: str = "ROBOFLOW_API_KEY",
) -> Path:
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing {api_key_env}. Set it to download Roboflow datasets.")

    from roboflow import Roboflow  # type: ignore

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    ds = rf.workspace(workspace).project(project).version(version).download(
        model_format=model_format,
        location=str(out_dir),
    )

    root = Path(ds.location)
    candidates = [root] + [p for p in root.iterdir() if p.is_dir()]
    for c in candidates:
        if any((c / s).exists() for s in ["train", "valid", "test"]):
            return c

    return root

def load_roboflow_split_as_imagefolder(root: str | Path, split: str) -> ImageFolder:
    """
    Roboflow "folder" exports usually contain train/valid/test folders.
    This loads one split with torchvision ImageFolder.
    """
    root = Path(root)
    split_dir = root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Roboflow split not found: {split_dir}")
    return ImageFolder(str(split_dir))