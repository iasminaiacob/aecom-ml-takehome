from __future__ import annotations
import argparse
import shutil
from pathlib import Path
import pandas as pd
from src.utils.config import ensure_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", type=str, required=True)
    parser.add_argument("--n", type=int, default=15)
    args = parser.parse_args()

    df = pd.read_csv(args.pred_csv)
    df = df[df["image_path"].notna() & (df["image_path"] != "")]
    df_err = df[df["true_class_id"] != df["pred_class_id"]]
    df_err = df_err.sort_values("confidence", ascending=False).head(args.n)

    out_dir = ensure_dir(Path(args.pred_csv).parent / "failure_gallery")

    lines = []
    for i, row in df_err.iterrows():
        src = Path(row["image_path"])
        dst = out_dir / src.name
        if src.exists():
            shutil.copy(src, dst)

        lines.append(
            f"- **{src.name}** | true: `{row['true_class_name']}` → "
            f"pred: `{row['pred_class_name']}` "
            f"(conf={row['confidence']:.2f})\n"
            f"  - Diagnosis: visually similar ingredients / texture bias\n"
        )

    readme = out_dir / "README.md"
    readme.write_text(
        "# Failure Gallery\n\n" + "\n".join(lines),
        encoding="utf-8",
    )

    print(f"Saved gallery: {out_dir}")

if __name__ == "__main__":
    main()