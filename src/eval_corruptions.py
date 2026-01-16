from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from src.datasets import load_food101
from src.models import build_model
from src.torch_datasets import HFDatasetWrapper
from src.transforms import build_eval_transforms
from src.utils.config import load_yaml, ensure_dir
from src.utils.corruptions import CORRUPTIONS
from src.utils.device import get_device
from src.metrics import accuracy_top1, macro_f1
from torch.utils.data import Subset

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class WildFolderDataset(Dataset):
    """
    Expects: root/class_name/*.jpg
    Label is derived from folder name using name2label mapping.
    """
    def __init__(self, root: Path, name2label: Dict[str, int], transform=None):
        self.root = root
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []

        for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
            cls = class_dir.name
            if cls not in name2label:
                continue
            y = int(name2label[cls])
            for p in class_dir.rglob("*"):
                if p.suffix.lower() in IMG_EXTS:
                    self.samples.append((p, y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, y

class CorruptedWrapper(Dataset):
    """
    Wrap a dataset that returns (PIL image, label) OR (tensor, label) depending on base.
    """
    def __init__(self, base: Dataset, corruption_name: str, severity: int, seed: int):
        self.base = base
        self.corruption_name = corruption_name
        self.severity = severity
        self.seed = seed
        self.fn = CORRUPTIONS[corruption_name]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        rng_seed = self.seed + idx * 1000 + self.severity * 10
        np.random.seed(rng_seed)
        x = self.fn(x, self.severity)
        return x, y

class PILHFDataset(Dataset):
    """HF dataset. Returns (PIL, label)"""
    def __init__(self, hf_split):
        self.hf_split = hf_split

    def __len__(self):
        return len(self.hf_split)

    def __getitem__(self, idx: int):
        ex = self.hf_split[int(idx)]
        return ex["image"].convert("RGB"), int(ex["label"])

class PILWildDataset(Dataset):
    """Wild folder dataset. Returns (PIL, label)"""
    def __init__(self, root: Path, name2label: Dict[str, int]):
        self.inner = WildFolderDataset(root, name2label, transform=None)

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx: int):
        p, y = self.inner.samples[idx]
        img = Image.open(p).convert("RGB")
        return img, y

@torch.no_grad()
def run_eval(model: torch.nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> Dict[str, float]:
    all_logits = []
    all_y = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x).detach().cpu().numpy()
        all_logits.append(logits)
        all_y.append(np.asarray(y))

    logits_np = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_y, axis=0)

    return {
        "acc1": float(accuracy_top1(logits_np, y_true)),
        "macro_f1": float(macro_f1(logits_np, y_true, num_classes=num_classes)),
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate corruption robustness on val and wild")
    parser.add_argument("--config", type=str, default="configs/train_easy.yaml")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_val_samples", type=int, default=5000)
    parser.add_argument("--max_wild_samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    run_name = cfg["project"]["run_name"]
    out_dir = ensure_dir(Path(cfg["project"]["output_dir"]) / run_name)

    device = get_device(args.device)
    print("Device: ", device)

    #load Food101 mappings
    hf_cache_dir = cfg["data"].get("hf_cache_dir", None)
    food = load_food101(hf_cache_dir=hf_cache_dir)
    ds = food.ds
    name2label = food.name2label

    #base PIL datasets so that corruptions happen in PIL space
    val_pil = PILHFDataset(ds["validation"])
    wild_root = Path("data/wild_images")
    wild_pil = PILWildDataset(wild_root, name2label)

    #subsample for speed
    rng = np.random.default_rng(args.seed)
    val_idx = np.arange(len(val_pil))
    rng.shuffle(val_idx)
    val_keep = val_idx[: min(args.max_val_samples, len(val_idx))]

    wild_idx = np.arange(len(wild_pil))
    rng.shuffle(wild_idx)
    wild_keep = wild_idx[: min(args.max_wild_samples, len(wild_idx))]

    #wrap with subset
    val_pil = Subset(val_pil, val_keep.tolist())
    wild_pil = Subset(wild_pil, wild_keep.tolist())

    #transforms after corruption
    image_size = int(cfg["data"]["image_size"])
    tfm = build_eval_transforms(image_size=image_size)

    #load model
    ckpt: Dict[str, Any] = torch.load(args.weights, map_location="cpu")
    model_name = ckpt["model_name"]
    num_classes = int(ckpt["num_classes"])

    model = build_model(model_name=model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    def make_loader(base_pil_ds: Dataset) -> DataLoader:
        #apply tensor transform
        class _TFM(Dataset):
            def __init__(self, inner: Dataset):
                self.inner = inner
            def __len__(self):
                return len(self.inner)
            def __getitem__(self, idx: int):
                img, y = self.inner[idx]
                return tfm(img), y

        return DataLoader(_TFM(base_pil_ds), batch_size=args.batch_size, shuffle=False, num_workers=0)

    rows = []

    splits = [("val", val_pil), ("wild", wild_pil)]

    for split_name, base_ds in splits:
        #clean
        loader = make_loader(base_ds)
        m_clean = run_eval(model, loader, device, num_classes=num_classes)
        rows.append({"split": split_name, "corruption": "clean", "severity": 0, **m_clean})

        for cname in CORRUPTIONS.keys():
            for severity in [1, 2, 3, 4, 5]:
                corr_ds = CorruptedWrapper(base_ds, cname, severity, seed=args.seed)
                loader = make_loader(corr_ds)
                m = run_eval(model, loader, device, num_classes=num_classes)
                rows.append({"split": split_name, "corruption": cname, "severity": severity, **m})
                print(split_name, cname, severity, m)

    df = pd.DataFrame(rows)
    out_csv = out_dir / "corruptions.csv"
    df.to_csv(out_csv, index=False)
    print("Wrote: ", out_csv)

if __name__ == "__main__":
    main()