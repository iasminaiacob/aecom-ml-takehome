from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
from pathlib import Path
import yaml
from torch.utils.data import Dataset, Subset

def load_label_map(path: str | Path) -> Dict[str, str]:
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    #normalize keys/values
    out = {}
    for k, v in d.items():
        out[str(k).strip().lower()] = str(v).strip().lower()
    return out

@dataclass(frozen=True)
class Harmonized:
    dataset: Dataset
    kept_indices: List[int]
    mapped_food_ids: List[int]  #same length as kept_indices

class HarmonizedSubset(Dataset):
    """
    Dataset wrapper: uses subset indices but returns (image, food101_label_id)
    """
    def __init__(self, base: Dataset, indices: List[int], labels: List[int]):
        assert len(indices) == len(labels)
        self.base = base
        self.indices = indices
        self.labels = labels

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        img, _orig_label = self.base[self.indices[i]]
        return img, int(self.labels[i])