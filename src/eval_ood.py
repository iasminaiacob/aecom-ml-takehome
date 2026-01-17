from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from src.datasets import load_food101
from src.models import build_model
from src.torch_datasets import HFDatasetWrapper
from src.transforms import build_eval_transforms
from src.utils.config import load_yaml, ensure_dir
from src.utils.seed import seed_everything
from src.ood_scores import (
    fit_mahalanobis,
    ood_score_energy,
    ood_score_mahalanobis,
    ood_score_msp,
)
from src.roboflow_ood import download_roboflow_classification_dataset, load_roboflow_split_as_imagefolder
from src.harmonize import load_label_map, HarmonizedSubset

@torch.no_grad()
def collect_logits(model, loader, device) -> np.ndarray:
    model.eval()
    outs = []
    for x, _ in loader:
        x = x.to(device)
        logits = model(x)
        outs.append(logits.detach().cpu().numpy())
    return np.concatenate(outs, axis=0)

def _maybe_subset(n: int, max_n: int) -> np.ndarray:
    idx = np.arange(n)
    if max_n is not None and max_n > 0 and n > max_n:
        return np.random.choice(idx, size=max_n, replace=False)
    return idx

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--label_map", default="configs/label_map.yaml")
    ap.add_argument(
    "--ood_method",
    type=str,
    default=None,
    choices=["msp", "energy", "mahalanobis"],
    help="Override OOD method from config (msp | energy | mahalanobis)",
)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    #reproducibility (prefer eval.seed, fallback to train.seed)
    if "eval" in cfg and "seed" in cfg["eval"]:
        seed = int(cfg["eval"]["seed"])
        deterministic = bool(cfg["eval"].get("deterministic", True))
    elif "train" in cfg:
        seed = int(cfg["train"]["seed"])
        deterministic = bool(cfg["train"]["deterministic"])
    else:
        seed = 1337
        deterministic = True

    seed_everything(seed, deterministic=deterministic)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"Device: {device}")

    num_classes = int(cfg["data"]["num_classes"])
    image_size = int(cfg["data"]["image_size"])
    batch_size = int(cfg["data"]["batch_size"])
    num_workers = int(cfg["data"]["num_workers"])
    pin_memory = bool(cfg["data"]["pin_memory"])

    #load model
    #model name can come from eval config or train config
    if "eval" in cfg and "model_name" in cfg["eval"]:
        model_name = cfg["eval"]["model_name"]
    elif "train" in cfg and "model_name" in cfg["train"]:
        model_name = cfg["train"]["model_name"]
    else:
        raise KeyError("Missing model_name. Add eval.model_name (preferred) or train.model_name in config.")

    model = build_model(str(model_name), num_classes=num_classes, pretrained=False)

    ckpt = torch.load(args.weights, map_location="cpu")

    #if it's a checkpoint dict, extract the real weights
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    #handle DataParallel "module." prefixes
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    # transforms
    tfm = build_eval_transforms(image_size=image_size)

    #Food101 val
    hf_cache_dir = cfg["data"].get("hf_cache_dir", None)
    food = load_food101(hf_cache_dir=hf_cache_dir)
    ds_val = food.ds["validation"]

    val_idx = _maybe_subset(len(ds_val), int(cfg["eval"]["max_val_samples"]))
    ds_val_sub = Subset(HFDatasetWrapper(ds_val, transform=tfm), val_idx.tolist())

    val_loader = DataLoader(
        ds_val_sub,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    #Wikimedia wild OOD
    wild_root = Path(cfg["eval"]["wild_root"])
    wild_loader = None
    if wild_root.exists():
        wild_ds = ImageFolder(str(wild_root), transform=tfm)
        wild_idx = _maybe_subset(len(wild_ds), int(cfg["eval"]["max_wild_samples"]))
        wild_ds_sub = Subset(wild_ds, wild_idx.tolist())
        wild_loader = DataLoader(
            wild_ds_sub,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        print("Wild root not found; skipping Wikimedia OOD set.")

    #Roboflow OOD
    rf_loader = None
    if bool(cfg["eval"].get("use_roboflow", False)):
        try:
            download_dir = Path(cfg["eval"]["roboflow_download_dir"])
            if download_dir.exists() and any((download_dir / s).exists() for s in ["train", "valid", "test"]):
                rf_dir = download_dir
            else:
                rf_dir = download_roboflow_classification_dataset(
                    out_dir=cfg["eval"]["roboflow_download_dir"],
                    workspace=cfg["eval"]["roboflow_workspace"],
                    project=cfg["eval"]["roboflow_project"],
                    version=int(cfg["eval"]["roboflow_version"]),
                )
            rf_split = cfg["eval"]["roboflow_split"]
            rf_ds = load_roboflow_split_as_imagefolder(rf_dir, rf_split)
            rf_ds.transform = tfm

            #label harmonization
            print("[roboflow] raw classes:", rf_ds.classes[:50], "..." if len(rf_ds.classes) > 50 else "")
            label_map = load_label_map(args.label_map)

            #Food-101 name -> id mapping
            food_names = list(food.ds["train"].features["label"].names)
            food_name_to_id = {name.lower(): i for i, name in enumerate(food_names)}

            #Roboflow ImageFolder class names
            rf_classes = [c.lower() for c in rf_ds.classes]  # folder names
            rf_class_to_idx = {c: i for i, c in enumerate(rf_classes)}

            kept_indices = []
            mapped_food_ids = []

            #rf_ds.samples is list of (path, class_idx)
            for i, (_path, class_idx) in enumerate(rf_ds.samples):
                ext_name = rf_classes[class_idx].strip().lower()

                if ext_name not in label_map:
                    continue

                food_name = label_map[ext_name]
                if food_name not in food_name_to_id:
                    continue

                kept_indices.append(i)
                mapped_food_ids.append(food_name_to_id[food_name])

            if len(kept_indices) == 0:
                raise RuntimeError(
                    "After label harmonization, no Roboflow samples remained. "
                    "Update configs/label_map.yaml to match Roboflow folder names."
                )

            rf_ds_h = HarmonizedSubset(rf_ds, kept_indices, mapped_food_ids)

            rf_loader = DataLoader(
                rf_ds_h,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

            print(f"[roboflow] kept {len(rf_ds_h)} / {len(rf_ds)} samples after harmonization")

        except Exception as e:
            print(f"Roboflow download/load skipped: {e}")

    #score computation
    # resolve OOD method priority: CLI > config > default
    if args.ood_method is not None:
        method = args.ood_method.lower()
    elif "ood" in cfg and "method" in cfg["ood"]:
        method = cfg["ood"]["method"].lower()
    else:
        method = "msp"

    print(f"[ood] using method = {method}")

    out_dir = ensure_dir(Path("outputs") / cfg["project"]["run_name"])
    results_path = out_dir / f"ood_results_{method}.json"

    #ID labels: 0; OOD labels: 1
    id_logits = collect_logits(model, val_loader, device=device)
    id_scores = None

    if method == "msp":
        id_scores = ood_score_msp(torch.tensor(id_logits)).numpy()
    elif method == "energy":
        t = float(cfg["ood"].get("temperature", 1.0))
        id_scores = ood_score_energy(torch.tensor(id_logits), temperature=t).numpy()
    elif method == "mahalanobis":
        #fit on ID (val subset) features
        stats = fit_mahalanobis(
            model=model,
            loader=val_loader,
            num_classes=num_classes,
            device=device,
            reg=float(cfg["ood"]["mahalanobis"]["reg"]),
            max_samples=int(cfg["ood"]["mahalanobis"]["max_fit_samples"]),
        )
        id_scores = ood_score_mahalanobis(model, val_loader, stats, device=device)
    else:
        raise ValueError(f"Unknown ood.method: {method}")

    def eval_one(name: str, ood_loader) -> Dict[str, float]:
        if ood_loader is None:
            return {}
        if method == "msp":
            ood_logits = collect_logits(model, ood_loader, device=device)
            ood_scores = ood_score_msp(torch.tensor(ood_logits)).numpy()
        elif method == "energy":
            ood_logits = collect_logits(model, ood_loader, device=device)
            t = float(cfg["ood"].get("temperature", 1.0))
            ood_scores = ood_score_energy(torch.tensor(ood_logits), temperature=t).numpy()
        else:
            #mahalanobis: reuse fitted stats
            ood_scores = ood_score_mahalanobis(model, ood_loader, stats, device=device)

        y_true = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
        scores = np.concatenate([id_scores, ood_scores])

        auroc = float(roc_auc_score(y_true, scores))
        auprc = float(average_precision_score(y_true, scores))
        return {"auroc": auroc, "auprc": auprc, "n_id": int(len(id_scores)), "n_ood": int(len(ood_scores))}

    results = {
        "method": method,
        "weights": args.weights,
        "id": {"split": "food101_validation", "n": int(len(id_scores))},
        "wild": eval_one("wild", wild_loader),
        "roboflow": eval_one("roboflow", rf_loader),
    }

    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote: {results_path}")
    print(results)

if __name__ == "__main__":
    main()