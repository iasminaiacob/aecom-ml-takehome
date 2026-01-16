from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from src.utils.config import ensure_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", type=str, required=True)
    parser.add_argument("--top_n", type=int, default=10)
    args = parser.parse_args()

    df = pd.read_csv(args.pred_csv)
    df_err = df[df["true_class_id"] != df["pred_class_id"]]

    pairs = (
        df_err.groupby(["true_class_name", "pred_class_name"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(args.top_n)
    )

    out_dir = ensure_dir(Path(args.pred_csv).parent)
    out_path = out_dir / "top_confusions.csv"
    pairs.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(pairs)

if __name__ == "__main__":
    main()