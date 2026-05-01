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

### Organise the processed images

The project expects the **processed images** (not raw DICOM) organised as:

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
|---|---|---|---|---|---|
| Train | 110 | 505 | 197 | 255 | 1,067 |
| Val | 24 | 98 | 59 | 53 | 234 |
| Test | 17 | 123 | 43 | 84 | 267 |

### If the dataset comes unsplit

If the downloaded archive has all images together with a `metadata.csv`, use the official split notebook from:
https://github.com/monicaccmartins/MIQR-CC-Dataset/blob/main/training/split_dataset.ipynb

Or adapt `scripts/run_eda.py` to create your own stratified split maintaining the same ratios.

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
  device: "cuda"        # "cpu" if no GPU
  batch_size: 32        # reduce to 16 if GPU OOM
  num_epochs: 30
  learning_rate: 0.0001

model:
  name: "efficientnet_b0"   # see supported models list

augmentation:
  use_clahe: false      # set true to enable CLAHE
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

### Option B: Train a specific model

```bash
# EfficientNet-B0 (recommended)
python scripts/train_experiment.py --model efficientnet_b0

# DenseNet121 with Focal Loss
python scripts/train_experiment.py --model densenet121 --loss focal

# EfficientNet-B0 with CLAHE preprocessing
python scripts/train_experiment.py --model efficientnet_b0 --clahe

# EfficientNet-B0 with WeightedRandomSampler
python scripts/train_experiment.py --model efficientnet_b0 --sampler
```

### Option C: Run all planned experiments (E0–E5)

```bash
python scripts/run_all_experiments.py
```

This runs all 6 experiments sequentially and writes a summary table to:
`outputs/tables/all_experiments_summary.csv`

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

All scripts accept `--config` to override the config file. To exactly reproduce an experiment:

```bash
# Reproduce E2 (EfficientNet-B0, no CLAHE, CE weighted)
python scripts/train_experiment.py \
  --model efficientnet_b0 \
  --loss weighted_cross_entropy \
  --tag E2 \
  --config config.yaml
```

The seed is fixed to `42` by default in `config.yaml`. Change it there to test variance.

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

| Name | Source | Notes |
|---|---|---|
| `resnet18` | torchvision | Baseline |
| `resnet50` | torchvision | Heavier baseline |
| `efficientnet_b0` | timm | Main recommended model |
| `efficientnet_b3` | timm | Larger variant |
| `densenet121` | torchvision | Good for medical imaging |
| `mobilenet_v3` | torchvision | Lightweight |
| `deit_tiny` | timm | Vision Transformer |

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `CUDA out of memory` | Reduce `batch_size` in config.yaml to 16 or 8 |
| `No module named 'timm'` | `pip install timm` |
| `No module named 'grad_cam'` | `pip install grad-cam` |
| `No images found` | Check dataset path in config.yaml and data/processed/ structure |
| `RuntimeError: expected scalar type` | Ensure images are RGB (3-channel) — the Dataset handles this automatically |
| Windows multiprocessing error | Set `num_workers: 0` in config.yaml |

---

## Hardware recommendations

| Setup | Expected time per epoch (1067 train images, bs=32) |
|---|---|
| NVIDIA RTX 3060+ | ~15–30 seconds |
| NVIDIA T4 (Colab) | ~20–40 seconds |
| CPU (i7/i9) | ~3–8 minutes |

Full training (30 epochs, EfficientNet-B0):
- GPU: ~10–15 minutes
- CPU: ~3–4 hours

---

## Citation

If using the MIQR-CC dataset:

```
Martins, M. et al. "Curated endoscopic retrograde cholangiopancreatography images dataset (MIQR-CC)"
figshare. Dataset. https://doi.org/10.6084/m9.figshare.31079236
```
