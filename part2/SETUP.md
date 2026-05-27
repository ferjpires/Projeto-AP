# SETUP.md — Complete Setup & Reproduction Guide

## MIQR-CC ERCP Classification Project

**Universidade do Minho | Aprendizagem Profunda | 2025/2026**

---

## Requirements

- Python 3.10 or 3.11
- pip ≥ 23
- At least 8 GB RAM
- GPU strongly recommended (NVIDIA with CUDA ≥ 11.8); CPU works but is slow

---

## Step 1 — Clone or extract the project

If from a zip file:

```bash
unzip MIQR-CC-AP-Project.zip
cd MIQR-CC-AP-Project
```

If from Git:

```bash
git clone <your-repo-url>
cd MIQR-CC-AP-Project
```

---

## Step 2 — Create Python environment

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (CMD):**

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

---

## Step 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### PyTorch with CUDA (recommended, if you have an NVIDIA GPU)

Check your CUDA version first: `nvidia-smi`

Then install the matching PyTorch build from https://pytorch.org/get-started/locally/

Example for CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Example for CUDA 11.8:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

CPU only (slow):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Then install the rest:

```bash
pip install -r requirements.txt
```

---

## Step 4 — Download and organise the dataset

### Download

The MIQR-CC dataset is available at:

- **Direct download:** https://figshare.com/ndownloader/files/61063177
- **Dataset page:** https://doi.org/10.6084/m9.figshare.31079236

This downloads a ZIP/archive. Extract it locally.

### Organise the processed images with the official split

Place the full processed image set in `data/raw/`, place `metadata.csv` at
`data/metadata.csv`, and place the official split from the reference repository
under `data/official_split/`:

```
data/
├── raw/                         # processed PNG images, flat directory
├── metadata.csv
└── official_split/
    ├── train/
    ├── val/
    └── test/
```

Then build the training directory:

```bash
python scripts/prepare_dataset.py
```

The script updates `data/metadata.csv` with official split columns and rebuilds
`data/processed/` as:

```
data/
└── processed/
    ├── train/
    │   ├── Biliary_Leaks/    ← PNG images
    │   ├── Lithiasis/
    │   ├── Normal/
    │   └── Stricture/
    ├── val/
    │   ├── Biliary_Leaks/
    │   ├── Lithiasis/
    │   ├── Normal/
    │   └── Stricture/
    └── test/
        ├── Biliary_Leaks/
        ├── Lithiasis/
        ├── Normal/
        └── Stricture/
```

### Expected image counts (from paper/official splits)

| Split | Biliary_Leaks | Lithiasis | Normal | Stricture | Total |
| ----- | ------------- | --------- | ------ | --------- | ----- |
| Train | 110           | 505       | 197    | 255       | 1,067 |
| Val   | 24            | 98        | 59     | 53        | 234   |
| Test  | 17            | 123       | 43     | 84        | 267   |

### If the dataset comes unsplit

Do not create a new stratified split for the final experiments. Use the fixed
split from the official repository so the test results are comparable with the
published baseline:
https://github.com/monicaccmartins/MIQR-CC-Dataset/tree/main/training/dataset

---

## Step 5 — Verify setup

```bash
python scripts/run_eda.py
```

This should print class counts for train/val/test and create figures in `outputs/figures/`.

If you see "No training images found", the dataset is not yet in place (see Step 4).

---

## Step 6 — Configure (optional)

Edit `config.yaml` to change:

```yaml
training:
  device: "cuda" # "cpu" if no GPU
  batch_size: 32 # reduce to 16 if GPU OOM
  num_epochs: 30
  learning_rate: 0.0001

model:
  name: "efficientnet_b0" # see supported models list

augmentation:
  use_clahe: false # set true to enable CLAHE
```

---

## Step 7 — Run experiments

### Option A: Train baseline only

```bash
python scripts/train_baseline.py
```

Trains ResNet18 with weighted CrossEntropy. Saves:

- `models/baseline_resnet18.pth`
- `outputs/figures/resnet18_baseline_curves.png`
- `outputs/confusion_matrices/resnet18_baseline_cm.png`
- `outputs/tables/resnet18_baseline_metrics.csv`

### Option B: Train a specific experiment

```bash
# E0 — ResNet18 + CLAHE + weighted CE
python scripts/train_experiment.py --model resnet18 --clahe --loss weighted_cross_entropy --tag E0

# E1 — EfficientNet-B0 + CLAHE + weighted CE
python scripts/train_experiment.py --model efficientnet_b0 --clahe --loss weighted_cross_entropy --tag E1

# E2 — DenseNet121 + Focal (ensemble feeder, run with 5 seeds)
for s in 42 123 777 7 2024; do
  python scripts/train_experiment.py --model densenet121 --loss focal --tag E2 --seed $s
done

# E3 — EfficientNet-B0 + weighted CE + WeightedRandomSampler
python scripts/train_experiment.py --model efficientnet_b0 --loss weighted_cross_entropy --sampler --tag E3

# E4 — ConvNeXt-tiny + Focal (ensemble feeder, run with 5 seeds)
for s in 42 123 777 7 2024; do
  python scripts/train_experiment.py --model convnext_tiny --loss focal --tag E4 --seed $s
done

# E5 — DeiT-tiny (Vision Transformer) + Focal
python scripts/train_experiment.py --model deit_tiny --loss focal --tag E5
```

### Option C: Mixed-architecture ensemble (E6 — final reported result)

```bash
python scripts/eval_ensemble.py \
    --checkpoints \
        densenet121:models/densenet121_focal_E2_seed42.pth \
        densenet121:models/densenet121_focal_E2_seed123.pth \
        densenet121:models/densenet121_focal_E2_seed777.pth \
        densenet121:models/densenet121_focal_E2_seed7.pth \
        densenet121:models/densenet121_focal_E2_seed2024.pth \
        convnext_tiny:models/convnext_tiny_focal_E4_seed42.pth \
        convnext_tiny:models/convnext_tiny_focal_E4_seed123.pth \
        convnext_tiny:models/convnext_tiny_focal_E4_seed777.pth \
        convnext_tiny:models/convnext_tiny_focal_E4_seed7.pth \
        convnext_tiny:models/convnext_tiny_focal_E4_seed2024.pth \
    --tta none --tag E6_ensemble_mixed
```

### Option D: Run everything end-to-end (E0–E6)

```bash
python scripts/run_all_experiments.py
```

This trains E0, E1, E3, E5 (one seed each), E2 and E4 (five seeds each), then
runs the E6 ensemble and writes `outputs/tables/all_experiments_summary.csv`.

**Smoke test (5 epochs each, ~minutes):**

```bash
python scripts/run_all_experiments.py --quick
```

---

## Step 8 — Evaluate a checkpoint

```bash
python scripts/evaluate_model.py --checkpoint models/baseline_resnet18.pth

# Evaluate on val instead of test
python scripts/evaluate_model.py --checkpoint models/baseline_resnet18.pth --split val
```

Outputs: classification report, confusion matrix, ROC curve, metrics CSV.

---

## Step 9 — Generate Grad-CAM

```bash
python scripts/generate_gradcam.py --checkpoint models/best_model.pth --n 3
```

Args:

- `--n`: number of examples per class (default 3)
- `--split`: which split to pull images from (default: test)

Saves visualisations to `outputs/gradcam/<class>/`.

Alternatively, open `notebooks/03_gradcam_examples.ipynb` for interactive exploration.

---

## Step 10 — Explore interactively (Jupyter)

```bash
jupyter notebook
```

Open notebooks in order:

1. `notebooks/01_eda.ipynb` — Class distributions, image sizes, sample images
2. `notebooks/02_check_images.ipynb` — CLAHE effect, augmentation preview
3. `notebooks/03_gradcam_examples.ipynb` — Grad-CAM visualisation

---

## Reproducing specific results

All scripts accept `--config` to override the config file. The seed is fixed
to `42` by default in `config.yaml`; pass `--seed N` to override per-run.

```bash
# Reproduce E4 single seed (ConvNeXt-tiny + Focal, seed 42)
python scripts/train_experiment.py \
  --model convnext_tiny --loss focal --tag E4 --seed 42

# Reproduce E6 final ensemble from existing checkpoints (no retraining)
python scripts/eval_ensemble.py \
  --checkpoints \
    densenet121:models/densenet121_focal_E2_seed42.pth \
    densenet121:models/densenet121_focal_E2_seed123.pth \
    densenet121:models/densenet121_focal_E2_seed777.pth \
    densenet121:models/densenet121_focal_E2_seed7.pth \
    densenet121:models/densenet121_focal_E2_seed2024.pth \
    convnext_tiny:models/convnext_tiny_focal_E4_seed42.pth \
    convnext_tiny:models/convnext_tiny_focal_E4_seed123.pth \
    convnext_tiny:models/convnext_tiny_focal_E4_seed777.pth \
    convnext_tiny:models/convnext_tiny_focal_E4_seed7.pth \
    convnext_tiny:models/convnext_tiny_focal_E4_seed2024.pth \
  --tta none --tag E6_ensemble_mixed
```

---

## Output files reference

After full run, expect:

```
outputs/
├── figures/
│   ├── class_distribution_train.png
│   ├── class_distribution_val.png
│   ├── class_distribution_test.png
│   ├── image_size_distribution.png
│   ├── sample_images_per_class.png
│   ├── resnet18_baseline_curves.png
│   └── efficientnet_b0_*_curves.png
├── confusion_matrices/
│   └── *_cm.png
├── roc_curves/
│   └── *_roc.png
├── gradcam/
│   ├── Biliary_Leaks/
│   ├── Lithiasis/
│   ├── Normal/
│   └── Stricture/
└── tables/
    ├── *_metrics.csv
    ├── *_classification_report.txt
    └── all_experiments_summary.csv
models/
├── baseline_resnet18.pth
└── efficientnet_b0_*.pth
```

---

## Supported architectures

| Name              | Source      | Notes                                                        |
| ----------------- | ----------- | ------------------------------------------------------------ |
| `resnet18`        | torchvision | Used in E0                                                   |
| `resnet50`        | torchvision | Available; not in default suite                              |
| `efficientnet_b0` | timm        | Used in E1, E3                                               |
| `convnext_tiny`   | timm        | Used in E4 (ensemble feeder, best single CNN)                |
| `deit_tiny`       | timm        | Vision Transformer (E5); pos. embeddings interpolated to 320 |
| `efficientnet_b3` | timm        | Larger variant                                               |
| `densenet121`     | torchvision | Good for medical imaging                                     |
| `mobilenet_v3`    | torchvision | Lightweight                                                  |
| `deit_tiny`       | timm        | Vision Transformer                                           |

---

## Citation

If using the MIQR-CC dataset:

```
Martins, M. et al. "Curated endoscopic retrograde cholangiopancreatography images dataset (MIQR-CC)"
figshare. Dataset. https://doi.org/10.6084/m9.figshare.31079236
```
