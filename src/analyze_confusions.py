from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from src.utils.config import ensure_dir

def plot_confusion(cm: np.ndarray, class_names, title: str, out_path: Path, max_classes: int = 20):
    """
    Plot a reduced confusion matrix (top-k classes by support) for readability.
    """
    #select top classes by row sum
    support = cm.sum(axis=1)
    idx = np.argsort(-support)[:max_classes]
    cm_small = cm[np.ix_(idx, idx)]
    names = [class_names[i] for i in idx]

    cm_norm = cm_small / (cm_small.sum(axis=1, keepdims=True) + 1e-9)

    plt.figure(figsize=(10, 9))
    plt.imshow(cm_norm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    plt.xticks(range(len(names)), names, rotation=90)
    plt.yticks(range(len(names)), names)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--max_classes", type=int, default=20)
    args = parser.parse_args()

    df = pd.read_csv(args.pred_csv)
    out_dir = ensure_dir(Path(args.out_dir))

    y_true = df["true_class_id"].to_numpy()
    y_pred = df["pred_class_id"].to_numpy()

    class_ids = sorted(df["true_class_id"].unique().tolist())
    class_names = (
        df.drop_duplicates("true_class_id")
        .sort_values("true_class_id")["true_class_name"]
        .tolist()
    )

    cm = confusion_matrix(y_true, y_pred, labels=class_ids)

    split = df["split"].iloc[0]
    out_path = out_dir / f"confusion_{split}.png"

    plot_confusion(
        cm,
        class_names=class_names,
        title=f"{split} confusion matrix (top classes)",
        out_path=out_path,
        max_classes=args.max_classes,
    )

    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()