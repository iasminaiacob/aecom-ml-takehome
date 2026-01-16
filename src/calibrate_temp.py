from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.datasets import load_food101
from src.torch_datasets import HFDatasetWrapper
from src.transforms import build_eval_transforms
from src.models import build_model
from src.utils.config import load_yaml, ensure_dir
from src.utils.device import get_device
from src.utils.ece import ece_from_logits
from src.utils.calibrate import fit_temperature, apply_temperature

@torch.no_grad()
def collect_val_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    all_logits = []
    all_y = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x).detach().cpu().numpy()
        all_logits.append(logits)
        all_y.append(y.numpy())
    return np.concatenate(all_logits, axis=0), np.concatenate(all_y, axis=0)

def main() -> None:
    parser = argparse.ArgumentParser(description="Temperature scaling on validation set")
    parser.add_argument("--config", type=str, default="configs/train_easy.yaml")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_bins", type=int, default=15)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    run_name = cfg["project"]["run_name"]
    out_dir = ensure_dir(Path(cfg["project"]["output_dir"]) / run_name)

    device = get_device(args.device)
    print("Device:", device)

    #load dataset
    hf_cache_dir = cfg["data"].get("hf_cache_dir", None)
    food = load_food101(hf_cache_dir=hf_cache_dir)
    ds = food.ds

    image_size = int(cfg["data"]["image_size"])
    tfm = build_eval_transforms(image_size=image_size)

    val_ds = HFDatasetWrapper(ds["validation"], transform=tfm)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    #load model
    ckpt: Dict[str, Any] = torch.load(args.weights, map_location="cpu")
    model_name = ckpt["model_name"]
    num_classes = int(ckpt["num_classes"])

    model = build_model(model_name=model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    logits_np, y_np = collect_val_logits(model, val_loader, device=device)

    #ECE before
    ece_before = ece_from_logits(logits_np, y_np, n_bins=args.n_bins).ece

    #fit temperature using torch tensors
    logits_t = torch.from_numpy(logits_np).float()
    y_t = torch.from_numpy(y_np).long()
    res = fit_temperature(logits_t, y_t, device="cpu")

    #ECE after
    scaled_logits_np = apply_temperature(logits_np, res.temperature)
    ece_after = ece_from_logits(scaled_logits_np, y_np, n_bins=args.n_bins).ece

    out = {
        "run_name": run_name,
        "weights": str(args.weights),
        "n_bins": int(args.n_bins),
        "temperature": float(res.temperature),
        "nll_before": float(res.nll_before),
        "nll_after": float(res.nll_after),
        "ece_before": float(ece_before),
        "ece_after": float(ece_after),
    }

    out_path = out_dir / "temperature_scaling.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Wrote: ", out_path)
    print(out)

if __name__ == "__main__":
    main()