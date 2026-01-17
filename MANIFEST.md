# MANIFEST.md

This repository implements the Easy and Medium tracks of the assignment using the Food-101 dataset and a “wild” image set from Wikimedia for out-of-distribution evaluation.
All commands below are run from the repository root.

---

## 0) Environment Setup

### Option A — Local (virtual environment)

Create and activate a virtual environment:

python -m venv .venv

Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

macOS / Linux:
source .venv/bin/activate

Install dependencies:
pip install -r requirements.txt

Optional storage configuration:
Users with limited local disk space may optionally set HF_HOME, HF_DATASETS_CACHE, HUGGINGFACE_HUB_CACHE, or TRANSFORMERS_CACHE to point to an external disk. These variables are not required and default to Hugging Face’s standard locations if unset.
$env:HF_HOME="E:\hf_home"
$env:HF_DATASETS_CACHE="E:\hf_home\datasets"
$env:HUGGINGFACE_HUB_CACHE="E:\hf_home\hub"
$env:TRANSFORMERS_CACHE="E:\hf_home\transformers"

---

### Option B — Docker

Build the Docker image:
docker build -t food101-takehome .

---

## 1) Dataset & EDA (Food-101)

Run exploratory data analysis:
python -m src.eda_food101

Outputs written to:
outputs/baseline_resnet18/
├── EDA_SUMMARY.txt
├── eda_class_counts.csv
├── eda_imbalance_summary.csv
├── eda_resolutions_sampled.csv
├── eda_resolutions_summary.csv
└── figs/
    ├── train_class_counts_sorted.png
    └── resolutions_scatter_sampled.png

---

## 2) Baseline Model Training (ResNet-18)

Train ImageNet-pretrained ResNet-18 with cross-entropy loss:
python -m src.train --config configs/train_easy.yaml

Outputs:
models/easy2_ce_resnet18.pt
outputs/easy2_ce_resnet18/train_history.csv

---

## 3) Inference CLI

Run inference on a custom directory of images (CPU):
python cli.py --images_dir sample_images --output_csv predictions.csv --weights models/easy2_ce_resnet18.pt --batch_size 16 --device cpu

GPU inference:
python cli.py --images_dir sample_images --output_csv predictions.csv --weights models/easy2_ce_resnet18.pt --batch_size 16 --device cuda

---

## 4) “Wild” Dataset Builder (Wikimedia)

Fetch and download out-of-distribution images using the Wikimedia API:
python -m src.wild_builder --config configs/wild.yaml

Outputs:
data/wild_images/
├── wild_metadata.csv
└── <class_name>/
    └── *.jpg / *.png

Wild images are not used for training.

---

## 5) Robust Loss + Imbalance Handling

Train with Generalized Cross-Entropy (GCE) loss and effective-number class reweighting:
python -m src.train --config configs/train.yaml

Outputs:
models/medium5_gce_effnum_resnet18.pt
outputs/medium5_gce_effnum_resnet18/train_history.csv

---

## 6) Calibration via Temperature Scaling

Fit temperature on validation split and compute ECE:
python -m src.calibrate_temp --config configs/train_easy.yaml --weights models/easy2_ce_resnet18.pt --device cuda --batch_size 64 --n_bins 15

Generate reliability diagram (before/after temperature scaling):
python -m src.plot_calibration --config configs/train_easy.yaml --weights models/easy2_ce_resnet18.pt --device cuda

Outputs:
outputs/easy2_ce_resnet18/temperature_scaling.json
outputs/easy2_ce_resnet18/calibration_reliability.png

---

## 7) Corruption Robustness

Evaluate model under synthetic corruptions (validation + wild):
python -m src.eval_corruptions --config configs/train_easy.yaml --weights models/easy2_ce_resnet18.pt --device cuda --batch_size 64 --max_val_samples 5000 --max_wild_samples 300

Output:
outputs/easy2_ce_resnet18/corruptions.csv

---

## 8) Failure Analysis

### 8.1 Generate per-image predictions
python -m src.predict_dataset --config configs/train_easy.yaml --weights models/easy2_ce_resnet18.pt --device cuda --batch_size 64 --max_val 5000 --max_wild 300

Outputs:
outputs/easy2_ce_resnet18/predictions_val.csv
outputs/easy2_ce_resnet18/predictions_wild.csv

### 8.2 Confusion matrices
python -m src.analyze_confusions --pred_csv outputs/easy2_ce_resnet18/predictions_val.csv --out_dir outputs/easy2_ce_resnet18
python -m src.analyze_confusions --pred_csv outputs/easy2_ce_resnet18/predictions_wild.csv --out_dir outputs/easy2_ce_resnet18

Outputs:
outputs/easy2_ce_resnet18/confusion_val.png
outputs/easy2_ce_resnet18/confusion_wild.png

### 8.3 Top confused class pairs
python -m src.top_confusions --pred_csv outputs/easy2_ce_resnet18/predictions_val.csv --top_n 10

### 8.4 Failure gallery (wild)
python -m src.failure_gallery --pred_csv outputs/easy2_ce_resnet18/predictions_wild.csv --n 15

Outputs:
outputs/easy2_ce_resnet18/failure_gallery/
├── README.md
└── *.jpg

---

## 9) Optional Backbone Ablation — EfficientNet-B0

Train EfficientNet-B0:
python -m src.train --config configs/train_effnet.yaml

Calibrate:
python -m src.calibrate_temp --config configs/train_effnet.yaml --weights models/easy_effnetb0.pt --device cuda --batch_size 64 --n_bins 15

Corruptions:
python -m src.eval_corruptions --config configs/train_effnet.yaml --weights models/easy_effnetb0.pt --device cuda --batch_size 64 --max_val_samples 5000 --max_wild_samples 300

Outputs:
models/easy_effnetb0.pt
outputs/easy_effnetb0/train_history.csv
outputs/easy_effnetb0/temperature_scaling.json
outputs/easy_effnetb0/corruptions.csv

---

## 10) Docker — CLI Example

Windows CMD:

docker run --rm ^
  -v %cd%\sample_images:/app/sample_images ^
  -v %cd%\models:/app/models ^
  -v %cd%:/app_out ^
  food101-takehome ^
  python cli.py --images_dir /app/sample_images --output_csv /app_out/predictions_docker.csv --weights /app/models/easy2_ce_resnet18.pt --batch_size 16 --device cpu

Windows Powershell:

docker run --rm `
  -v ${PWD}\sample_images:/app/sample_images `
  -v ${PWD}\models:/app/models `
  -v ${PWD}:/app_out `
  food101-takehome `
  python cli.py --images_dir /app/sample_images --output_csv /app_out/predictions_docker.csv --weights /app/models/easy2_ce_resnet18.pt --batch_size 16 --device cpu

(Note: Replace`^` line continuation with `\` on macOS/Linux.)

## 11) Tests

Run all unit tests:
pytest -q

## 12) D1 — Cross-Dataset Generalization & OOD Detection

This section evaluates out-of-distribution (OOD) detection when training on Food-101 and testing on external datasets.
Three OOD scoring methods are implemented: **MSP**, **Energy**, and **Mahalanobis distance**.

Set the Roboflow API key (required once):

Powershell:
$env:ROBOFLOW_API_KEY="YOUR_API_KEY"

Download the dataset in folder format:
roboflow download -f folder -l data/roboflow_ood iasminaiacob/fast-food-classification-ddqyc-u0ijs/1

The extracted directory is expected to contain:
data/roboflow_ood/
├── train/
├── valid/
└── test/

MSP:
python -m src.eval_ood --config configs/d1_ood.yaml --weights models/easy2_ce_resnet18.pt  --device cuda --label_map configs/label_map.yaml --ood_method msp

Energy:
python -m src.eval_ood --config configs/d1_ood.yaml --weights models/easy2_ce_resnet18.pt  --device cuda --label_map configs/label_map.yaml --ood_method energy

Mahalanobis:
python -m src.eval_ood --config configs/d1_ood.yaml --weights models/easy2_ce_resnet18.pt  --device cuda --label_map configs/label_map.yaml --ood_method mahalanobis