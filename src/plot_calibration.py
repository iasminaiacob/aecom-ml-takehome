from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.datasets import load_food101
from src.models import build_model
from src.torch_datasets import HFDatasetWrapper
from src.transforms import build_eval_transforms
from src.utils.calibrate import fit_temperature, apply_temperature
from src.utils.config import load_yaml, ensure_dir
from src.utils.device import get_device
from src.utils.ece import ece_from_logits, ECEResult

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

def _plot_reliability(ax, ece_res: ECEResult, label: str) -> None:
    """
    Plot per-bin accuracy vs confidence using ECE bin outputs.
    Uses bin mean confidence as x, bin accuracy as y.
    """
    x = ece_res.bin_conf
    y = ece_res.bin_acc
    mask = np.isfinite(x) & np.isfinite(y) #skip empty bins

    ax.plot(x[mask], y[mask], marker="o", linewidth=1.5, label=label)

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reliability diagram (before/after temp scaling).")
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

    #dataset / loader
    hf_cache_dir = cfg["data"].get("hf_cache_dir", None)
    food = load_food101(hf_cache_dir=hf_cache_dir)
    ds = food.ds

    image_size = int(cfg["data"]["image_size"])
    tfm = build_eval_transforms(image_size=image_size)

    val_ds = HFDatasetWrapper(ds["validation"], transform=tfm)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    #model
    ckpt: Dict[str, Any] = torch.load(args.weights, map_location="cpu")
    model = build_model(
        model_name=str(ckpt["model_name"]),
        num_classes=int(ckpt["num_classes"]),
        pretrained=False,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    #collect logits
    logits_np, y_np = collect_val_logits(model, val_loader, device=device)

    #ECE before
    ece_before = ece_from_logits(logits_np, y_np, n_bins=args.n_bins)

    #fit temperature on CPU
    logits_t = torch.from_numpy(logits_np).float()
    y_t = torch.from_numpy(y_np).long()
    ts_res = fit_temperature(logits_t, y_t, device="cpu")

    #ECE after
    scaled_logits_np = apply_temperature(logits_np, ts_res.temperature)
    ece_after = ece_from_logits(scaled_logits_np, y_np, n_bins=args.n_bins)

    #plot
    fig = plt.figure(figsize=(6.5, 6.0))
    ax = fig.add_subplot(111)

    #perfect calibration diagonal
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.0, label="perfect")

    _plot_reliability(ax, ece_before, label=f"before (ECE={ece_before.ece:.3f})")
    _plot_reliability(ax, ece_after, label=f"after (ECE={ece_after.ece:.3f}, T={ts_res.temperature:.3f})")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability diagram (validation)")
    ax.legend(loc="lower right")
    fig.tight_layout()

    out_path = out_dir / "calibration_reliability.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print("Wrote: ", out_path)
    print(f"Temperature: {ts_res.temperature:.6f}")
    print(f"NLL before: {ts_res.nll_before:.6f}  after: {ts_res.nll_after:.6f}")
    print(f"ECE before: {ece_before.ece:.6f}  after: {ece_after.ece:.6f}")

if __name__ == "__main__":
    main()