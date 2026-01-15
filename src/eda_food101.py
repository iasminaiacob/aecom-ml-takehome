from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, Iterable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.datasets import load_food101
from src.utils.config import load_yaml, ensure_dir
from src.utils.seed import seed_everything

def _class_counts(labels: Iterable[int], num_classes: int) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=np.int64)
    for y in labels:
        counts[int(y)] += 1
    return counts

def _imbalance_summary(counts: np.ndarray) -> Dict[str, float]:
    counts = counts.astype(np.float64)
    nonzero = counts[counts > 0]
    if len(nonzero) == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "cv": 0.0,
            "imbalance_ratio_max_over_min": math.inf,
        }

    mn = float(nonzero.min())
    mx = float(nonzero.max())
    mean = float(nonzero.mean())
    std = float(nonzero.std(ddof=0))
    cv = float(std / mean) if mean > 0 else 0.0
    irr = float(mx / mn) if mn > 0 else math.inf

    return {
        "min": mn,
        "max": mx,
        "mean": mean,
        "std": std,
        "cv": cv,
        "imbalance_ratio_max_over_min": irr,
    }

def _iter_resolutions(dataset, sample_size: int | None, rng: np.random.Generator):
    n = len(dataset)
    if sample_size is None or sample_size >= n:
        indices = range(n)
    else:
        indices = rng.choice(n, size=sample_size, replace=False)

    for i in indices:
        img = dataset[int(i)]["image"] #PIL Image
        w, h = img.size
        yield (w, h)

def main() -> None:
    cfg = load_yaml("configs/train.yaml")
    seed = int(cfg["train"]["seed"])
    deterministic = bool(cfg["train"]["deterministic"])
    seed_everything(seed, deterministic=deterministic)

    out_dir = ensure_dir(Path(cfg["project"]["output_dir"]) / cfg["project"]["run_name"])
    ensure_dir(out_dir / "figs")

    hf_cache_dir = cfg["data"].get("hf_cache_dir", None)

    #load dataset
    food = load_food101(hf_cache_dir=hf_cache_dir)
    ds = food.ds
    num_classes = int(cfg["data"]["num_classes"])

    #class counts
    splits = [k for k in ds.keys()]
    rows = []
    for split in splits:
        labels = ds[split]["label"]
        counts = _class_counts(labels, num_classes=num_classes)
        for cls_id, c in enumerate(counts.tolist()):
            rows.append(
                {
                    "split": split,
                    "class_id": cls_id,
                    "class_name": food.label2name[cls_id],
                    "count": int(c),
                }
            )

    df_counts = pd.DataFrame(rows)
    counts_path = out_dir / "eda_class_counts.csv"
    df_counts.to_csv(counts_path, index=False)

    #imbalance summary on train split
    train_counts = (
        df_counts[df_counts["split"] == "train"]
        .sort_values("class_id")["count"]
        .to_numpy(dtype=np.int64)
    )
    imb = _imbalance_summary(train_counts)
    df_imb = pd.DataFrame([imb])
    imb_path = out_dir / "eda_imbalance_summary.csv"
    df_imb.to_csv(imb_path, index=False)

    #resolution stats
    rng = np.random.default_rng(seed)

    res_rows = []
    for split in splits:
        sample_size = 5000 if split == "train" else 2000
        for (w, h) in _iter_resolutions(ds[split], sample_size=sample_size, rng=rng):
            res_rows.append({"split": split, "width": int(w), "height": int(h)})

    df_res = pd.DataFrame(res_rows)
    res_path = out_dir / "eda_resolutions_sampled.csv"
    df_res.to_csv(res_path, index=False)

    #summary table per split
    def summarize(group: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "n": len(group),
                "width_min": group["width"].min(),
                "width_p50": group["width"].median(),
                "width_p95": group["width"].quantile(0.95),
                "width_max": group["width"].max(),
                "height_min": group["height"].min(),
                "height_p50": group["height"].median(),
                "height_p95": group["height"].quantile(0.95),
                "height_max": group["height"].max(),
            }
        )

    summ_rows = []
    for split, g in df_res.groupby("split"):
        s = summarize(g).to_dict()
        s["split"] = split
        summ_rows.append(s)

    df_res_summary = pd.DataFrame(summ_rows)[
        ["split", "n", "width_min", "width_p50", "width_p95", "width_max",
        "height_min", "height_p50", "height_p95", "height_max"]
    ]

    res_sum_path = out_dir / "eda_resolutions_summary.csv"
    df_res_summary.to_csv(res_sum_path, index=False)

    #plots
    #train class count histogram
    df_train = df_counts[df_counts["split"] == "train"].sort_values("count", ascending=True)

    plt.figure()
    plt.plot(df_train["count"].to_numpy())
    plt.xlabel("Class (sorted by count)")
    plt.ylabel("Train images per class")
    plt.title("Food-101 sorted train class counts")
    plt.tight_layout()
    plt.savefig(out_dir / "figs" / "train_class_counts_sorted.png", dpi=200)
    plt.close()

    #resolution scatter
    plt.figure()
    plt.scatter(df_res["width"], df_res["height"], s=5)
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.title("Food-101 image resolutions (sampled)")
    plt.tight_layout()
    plt.savefig(out_dir / "figs" / "resolutions_scatter_sampled.png", dpi=200)
    plt.close()

    #summary text
    summary_txt = out_dir / "EDA_SUMMARY.txt"
    with summary_txt.open("w", encoding="utf-8") as f:
        f.write("Food-101 EDA outputs\n")
        f.write(f"- Counts: {counts_path}\n")
        f.write(f"- Imbalance summary (train): {imb_path}\n")
        f.write(f"- Resolutions (sampled): {res_path}\n")
        f.write(f"- Resolutions summary: {res_sum_path}\n")
        f.write("Figures:\n")
        f.write(f"- {out_dir / 'figs' / 'train_class_counts_sorted.png'}\n")
        f.write(f"- {out_dir / 'figs' / 'resolutions_scatter_sampled.png'}\n")
        f.write("\nImbalance summary (train):\n")
        for k, v in imb.items():
            f.write(f"{k}: {v}\n")

    print("Done. Wrote:")
    print(" -", counts_path)
    print(" -", imb_path)
    print(" -", res_path)
    print(" -", res_sum_path)
    print(" -", summary_txt)


if __name__ == "__main__":
    main()