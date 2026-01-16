import csv
from pathlib import Path
import subprocess
import sys

def test_cli_writes_predictions_csv(tmp_path: Path):
    """
    Smoke test: CLI produces a CSV with expected columns.
    Uses CPU to be CI-friendly.
    """
    repo_root = Path(__file__).resolve().parents[1]

    weights = repo_root / "models" / "easy2_ce_resnet18.pt"
    images_dir = repo_root / "sample_images"
    out_csv = tmp_path / "preds.csv"

    #skip if artifacts not present
    if not weights.exists() or not images_dir.exists():
        return

    cmd = [
        sys.executable,
        str(repo_root / "cli.py"),
        "--images_dir",
        str(images_dir),
        "--output_csv",
        str(out_csv),
        "--weights",
        str(weights),
        "--batch_size",
        "4",
        "--device",
        "cpu",
    ]
    subprocess.check_call(cmd)

    assert out_csv.exists()
    with out_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        expected = {"filename", "pred_class_id", "pred_class_name", "confidence"}
        assert expected.issubset(cols)

        rows = list(reader)
        assert len(rows) > 0