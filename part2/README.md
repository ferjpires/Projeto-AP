# MIQR-CC ERCP Classification — Deep Learning Project

**Universidade do Minho | Aprendizagem Profunda | 2025/2026**

Automatic classification of ERCP (Endoscopic Retrograde Cholangiopancreatography) fluoroscopic images into 4 diagnostic categories using Deep Learning, with interpretability via Grad-CAM.

---

## Classes

| Class | Description |
|---|---|
| `Biliary_Leaks` | Bile leaks / Fugas de bílis |
| `Lithiasis` | Biliary stones / Cálculos |
| `Stricture` | Biliary strictures / Estenoses |
| `Normal` | Normal findings |

## Baseline to beat
**F1 macro = 0.738** (from the MIQR-CC paper)

---

## Quick Start

```bash
# 1. Setup
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt

# 2. Place dataset (see SETUP.md for details)
# data/processed/train/<class>/*.png
# data/processed/val/<class>/*.png
# data/processed/test/<class>/*.png

# 3. EDA
python scripts/run_eda.py

# 4. Train baseline (ResNet18)
python scripts/train_baseline.py

# 5. Train best model (EfficientNet-B0)
python scripts/train_experiment.py --model efficientnet_b0

# 6. Evaluate
python scripts/evaluate_model.py --checkpoint models/efficientnet_b0_weighted_cross_entropy_E2.pth

# 7. Grad-CAM
python scripts/generate_gradcam.py --checkpoint models/best_model.pth

# 8. Run all experiments
python scripts/run_all_experiments.py
```

---

## Project Structure

```
MIQR-CC-AP-Project/
├── config.yaml                  # Central configuration
├── requirements.txt
├── SETUP.md                     # Full setup & reproduction guide
├── data/
│   ├── processed/               # ← Place dataset here (not in Git)
│   └── README.md
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis
│   ├── 02_check_images.ipynb   # CLAHE & augmentation preview
│   └── 03_gradcam_examples.ipynb
├── src/
│   ├── data/
│   │   ├── dataset.py           # ERCPDataset + DataLoaders
│   │   ├── transforms.py        # CLAHE + augmentation pipelines
│   │   └── eda.py               # EDA utilities
│   ├── models/
│   │   ├── build_model.py       # Model factory (ResNet/EfficientNet/DenseNet/DeiT)
│   │   └── losses.py            # CrossEntropy weighted + Focal Loss
│   ├── training/
│   │   ├── train.py             # Training loop + early stopping
│   │   ├── evaluate.py          # Inference + full test evaluation
│   │   └── metrics.py           # F1/AUC/confusion matrix
│   ├── explainability/
│   │   └── gradcam.py           # Grad-CAM for all model families
│   └── utils/
│       ├── seed.py              # Reproducibility seeds
│       ├── paths.py             # Config loading + dir management
│       └── plots.py             # All visualisation helpers
├── scripts/
│   ├── run_eda.py
│   ├── train_baseline.py        # ResNet18 baseline
│   ├── train_experiment.py      # Any model + hyperparams
│   ├── evaluate_model.py        # Evaluate any checkpoint
│   ├── generate_gradcam.py      # Grad-CAM for any checkpoint
│   └── run_all_experiments.py   # Run all 6 planned experiments
├── experiments/                 # Auto-created per experiment
├── outputs/
│   ├── figures/                 # EDA + learning curves
│   ├── confusion_matrices/
│   ├── roc_curves/
│   ├── gradcam/                 # Per-class Grad-CAM images
│   └── tables/                  # Metrics CSVs
└── models/                      # Best checkpoints
```

---

## Planned Experiments

| ID | Model | CLAHE | Loss | Sampler |
|---|---|---|---|---|
| E0 | ResNet18 | No | CE weighted | No |
| E1 | ResNet18 | Yes | CE weighted | No |
| E2 | EfficientNet-B0 | No | CE weighted | No |
| E3 | EfficientNet-B0 | Yes | CE weighted | No |
| E4 | DenseNet121 | No | Focal | No |
| E5 | EfficientNet-B0 | No | CE weighted | Yes |

---

## Supported Models

- `resnet18` — baseline
- `resnet50`
- `efficientnet_b0` — recommended main model
- `efficientnet_b3`
- `densenet121` — good for medical imaging
- `mobilenet_v3`
- `deit_tiny` — Vision Transformer (extra)

## Dataset

- **Paper:** https://doi.org/10.6084/m9.figshare.31079236
- **Download:** https://figshare.com/ndownloader/files/61063177
- **GitHub reference:** https://github.com/monicaccmartins/MIQR-CC-Dataset
- **Total:** 1,602 patients | 19,317 processed images | 5,519 labelled
