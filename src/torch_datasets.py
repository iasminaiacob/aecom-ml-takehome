from __future__ import annotations
from typing import Callable, Optional
from torch.utils.data import Dataset

class HFDatasetWrapper(Dataset):
    """
    Wraps a HuggingFace dataset split so it behaves like a PyTorch Dataset.
    """
    def __init__(self, hf_split, transform: Optional[Callable] = None):
        self.hf_split = hf_split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.hf_split)

    def __getitem__(self, idx: int):
        example = self.hf_split[int(idx)]
        img = example["image"]
        label = int(example["label"])

        try:
            #some PIL images can be truncated
            img.load()
        except Exception:
            #fallback: return a blank RGB image
            from PIL import Image
            img = Image.new("RGB", (224, 224), color=(0, 0, 0))

        if self.transform is not None:
            img = self.transform(img)

        return img, label