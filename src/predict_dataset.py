from __future__ import annotations
import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from src.datasets import load_food101
from src.models import build_model
from src.transforms import build_eval_transforms
from src.utils.config import load_yaml, ensure_dir
from src.utils.device import get_device

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

@dataclass
class PredRow:
    split: str
    image_id: str
    image_path: str
    true_class_id: int
    true_class_name: str
    pred_class_id: int
    pred_class_name: str
    confidence: float

class PILHFDataset(Dataset):
    """HF split. Returns (PIL, label, id_str)"""
    def __init__(self, hf_split):
        self.hf_split = hf_split

    def __len__(self) -> int:
        return len(self.hf_split)

    def __getitem__(self, idx: int):
        ex = self.hf_split[int(idx)]
        img = ex["image"].convert("RGB")
        y = int(ex["label"])
        #HF split doesn't give stable filename
        return img, y, f"hf_{idx:06d}"


class WildFolderDataset(Dataset):
    """
    Expects: root/class_name/*.jpg
    Label from folder name
    Returns (PIL, label, id_str, path)
    """
    def __init__(self, root: Path, name2label: Dict[str, int]):
        self.root = root
        self.name2label = name2label
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
        return img, int(y), path.stem, str(path)

class ApplyTransform(Dataset):
    """Wrap (PIL,label,..) dataset and apply tensor transform."""
    def __init__(self, base: Dataset, transform):
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        img = item[0]
        rest = item[1:]
        x = self.transform(img)
        return (x, *rest)

@torch.no_grad()
def predict_to_rows(
    split_name: str,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    label2name: Dict[int, str],
) -> List[PredRow]:
    rows: List[PredRow] = []

    for batch in loader:
        #batch: x, y, id, (optional path)
        x = batch[0].to(device, non_blocking=True)
        y = batch[1].cpu().numpy().astype(np.int64)
        image_ids = batch[2]
        image_paths = batch[3] if len(batch) > 3 else ["" for _ in range(len(image_ids))]

        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

        pred_np = pred.cpu().numpy().astype(np.int64)
        conf_np = conf.cpu().numpy().astype(np.float64)

        for i in range(len(image_ids)):
            true_id = int(y[i])
            pred_id = int(pred_np[i])
            rows.append(
                PredRow(
                    split=split_name,
                    image_id=str(image_ids[i]),
                    image_path=str(image_paths[i]),
                    true_class_id=true_id,
                    true_class_name=label2name[true_id],
                    pred_class_id=pred_id,
                    pred_class_name=label2name[pred_id],
                    confidence=float(conf_np[i]),
                )
            )

    return rows

def write_csv(path: Path, rows: List[PredRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate per-image predictions for val and wild")
    parser.add_argument("--config", type=str, default="configs/train_easy.yaml")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_val", type=int, default=5000)
    parser.add_argument("--max_wild", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    run_name = cfg["project"]["run_name"]
    out_dir = ensure_dir(Path(cfg["project"]["output_dir"]) / run_name)

    device = get_device(args.device)
    print("Device:", device)

    #dataset + label maps
    hf_cache_dir = cfg["data"].get("hf_cache_dir", None)
    food = load_food101(hf_cache_dir=hf_cache_dir)
    ds = food.ds
    label2name = food.label2name
    name2label = food.name2label

    #transforms
    image_size = int(cfg["data"]["image_size"])
    tfm = build_eval_transforms(image_size=image_size)

    rng = np.random.default_rng(args.seed)

    #val: subsample for speed
    val_base = PILHFDataset(ds["validation"])
    val_idx = np.arange(len(val_base))
    rng.shuffle(val_idx)
    val_keep = val_idx[: min(args.max_val, len(val_idx))].tolist()
    from torch.utils.data import Subset
    val_base = Subset(val_base, val_keep)
    val_ds = ApplyTransform(val_base, tfm)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    #wild: subsample
    wild_root = Path("data/wild_images")
    wild_base = WildFolderDataset(wild_root, name2label=name2label)
    wild_idx = np.arange(len(wild_base))
    rng.shuffle(wild_idx)
    wild_keep = wild_idx[: min(args.max_wild, len(wild_idx))].tolist()
    wild_base = Subset(wild_base, wild_keep)
    wild_ds = ApplyTransform(wild_base, tfm)
    wild_loader = DataLoader(wild_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    #load model
    ckpt: Dict[str, Any] = torch.load(args.weights, map_location="cpu")
    model = build_model(model_name=ckpt["model_name"], num_classes=int(ckpt["num_classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    #predict
    val_rows = predict_to_rows("val", model, val_loader, device, label2name)
    wild_rows = predict_to_rows("wild", model, wild_loader, device, label2name)

    val_csv = out_dir / "predictions_val.csv"
    wild_csv = out_dir / "predictions_wild.csv"

    write_csv(val_csv, val_rows)
    write_csv(wild_csv, wild_rows)

    print(f"Wrote {len(val_rows)} rows: {val_csv}")
    print(f"Wrote {len(wild_rows)} rows: {wild_csv}")

if __name__ == "__main__":
    main()