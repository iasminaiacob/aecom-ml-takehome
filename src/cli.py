from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from src.models import build_model
from src.utils.device import get_device

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

"""Loads saved checkpoint (models/)
Reads all images in a folder (jpg/png/jpeg/webp)
Runs batch inference on CPU or CUDA
Writes output_csv with: filename, predicted_class_id, predicted_class_name, confidence."""

class ImageFolderDataset(Dataset):
    def __init__(self, images: List[Path], tfm):
        self.images = images
        self.tfm = tfm

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        p = self.images[idx]
        img = Image.open(p).convert("RGB")
        x = self.tfm(img)
        return x, p.name

def build_infer_transforms(image_size: int):
    return transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Inference CLI for Food-101 classifier")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory with images")
    parser.add_argument("--output_csv", type=str, required=True, help="Where to write predictions.csv")
    parser.add_argument("--weights", type=str, required=True, help="Path to model checkpoint .pt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"--images_dir not found: {images_dir}")

    weight_path = Path(args.weights)
    if not weight_path.exists():
        raise FileNotFoundError(f"--weights not found: {weight_path}")

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    print("Device:", device)

    ckpt: Dict[str, Any] = torch.load(weight_path, map_location="cpu")
    model_name = ckpt["model_name"]
    num_classes = int(ckpt["num_classes"])
    image_size = int(ckpt.get("image_size", 224))
    label2name: Dict[int, str] = {int(k): v for k, v in ckpt["label2name"].items()} if isinstance(ckpt["label2name"], dict) else ckpt["label2name"]

    model = build_model(model_name=model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    tfm = build_infer_transforms(image_size=image_size)

    images = sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])
    if len(images) == 0:
        raise ValueError(f"No images found in {images_dir} (supported: {sorted(IMG_EXTS)})")

    ds = ImageFolderDataset(images, tfm)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    rows = []
    for x, names in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

        conf = conf.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()

        for fname, p, c in zip(names, pred, conf):
            p_int = int(p)
            rows.append({
                "filename": fname,
                "pred_class_id": p_int,
                "pred_class_name": label2name.get(p_int, str(p_int)),
                "confidence": float(c),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} predictions to {out_path}")

if __name__ == "__main__":
    main()