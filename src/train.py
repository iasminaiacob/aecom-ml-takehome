from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.datasets import load_food101
from src.metrics import accuracy_top1, macro_f1
from src.models import build_model
from src.torch_datasets import HFDatasetWrapper
from src.transforms import build_train_transforms, build_eval_transforms
from src.utils.config import load_yaml, ensure_dir
from src.utils.device import get_device
from src.utils.seed import seed_everything, seed_worker

@dataclass
class EpochResult:
    epoch: int
    split: str
    loss: float
    acc1: float
    macro_f1: float

@torch.no_grad()
def run_eval(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> Tuple[float, float, float]:
    model.eval()
    losses = []
    all_logits = []
    all_y = []

    criterion = nn.CrossEntropyLoss()

    for x, y in tqdm(loader, desc="eval", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        losses.append(loss.item())
        all_logits.append(logits.detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())

    logits_np = np.concatenate(all_logits, axis=0)
    y_np = np.concatenate(all_y, axis=0)

    return float(np.mean(losses)), accuracy_top1(logits_np, y_np), macro_f1(logits_np, y_np, num_classes=num_classes)

def main() -> None:
    cfg = load_yaml("configs/train.yaml")

    run_name = cfg["project"]["run_name"]
    out_dir = ensure_dir(Path(cfg["project"]["output_dir"]) / run_name)
    models_dir = ensure_dir("models")

    seed = int(cfg["train"]["seed"])
    deterministic = bool(cfg["train"]["deterministic"])
    seed_everything(seed, deterministic=deterministic)

    device = get_device(cfg["device"]["prefer"])
    print("Device:", device)

    #transforms
    image_size = int(cfg["data"]["image_size"])
    batch_size = int(cfg["data"]["batch_size"])
    num_workers = int(cfg["data"]["num_workers"])
    pin_memory = bool(cfg["data"]["pin_memory"])
    num_classes = int(cfg["data"]["num_classes"])
    hf_cache_dir = cfg["data"].get("hf_cache_dir", None)

    train_tfm = build_train_transforms(image_size=image_size, seed=seed)
    eval_tfm = build_eval_transforms(image_size=image_size)

    #load dataset
    food = load_food101(hf_cache_dir=hf_cache_dir)
    ds = food.ds

    #HF Food-101 provides train and validation by default
    train_ds = HFDatasetWrapper(ds["train"], transform=train_tfm)
    val_ds = HFDatasetWrapper(ds["validation"], transform=eval_tfm)

    #dataloader generator for determinism
    g = torch.Generator()
    g.manual_seed(seed)

    dl_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=g,
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **dl_kwargs)


    #model
    model = build_model(
        model_name=cfg["train"]["model_name"],
        num_classes=num_classes,
        pretrained=bool(cfg["train"]["pretrained"]),
    ).to(device)

    #optimizer, loss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    use_amp = bool(cfg["train"].get("amp", False)) and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    epochs = int(cfg["train"]["epochs"])
    best_val_f1 = -1.0
    history = []

    #train loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        max_steps = cfg["train"].get("max_steps_per_epoch", None)
        max_steps = int(max_steps) if max_steps is not None else None
        step = 0

        pbar = tqdm(train_loader, desc=f"train epoch {epoch}/{epochs}", leave=False)
        for x, y in pbar:
            
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())
            pbar.set_postfix(loss=float(np.mean(train_losses)))

            step += 1
            if max_steps is not None and step >= max_steps:
                break

        #eval
        train_loss = float(np.mean(train_losses))
        val_loss, val_acc1, val_f1 = run_eval(model, val_loader, device, num_classes=num_classes)

        history.append(EpochResult(epoch, "train", train_loss, float("nan"), float("nan")))
        history.append(EpochResult(epoch, "val", val_loss, val_acc1, val_f1))

        steps_info = f"{step}" if max_steps is None else f"{step}/{max_steps}"
        print(
            f"[epoch {epoch}] steps={steps_info} "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc1={val_acc1:.4f}  val_macro_f1={val_f1:.4f}"
        )

        #save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            ckpt_path = Path(models_dir) / f"{run_name}.pt"
            torch.save(
                {
                    "model_name": cfg["train"]["model_name"],
                    "num_classes": num_classes,
                    "state_dict": model.state_dict(),
                    "label2name": food.label2name,
                    "name2label": food.name2label,
                    "image_size": image_size,
                    "seed": seed,
                },
                ckpt_path,
            )
            print("Saved best model: ", ckpt_path)

    #save history CSV
    df = pd.DataFrame([r.__dict__ for r in history])
    hist_path = out_dir / "train_history.csv"
    df.to_csv(hist_path, index=False)
    print("Wrote history: ", hist_path)

if __name__ == "__main__":
    main()