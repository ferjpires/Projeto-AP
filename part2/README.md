# MIQR-CC ERCP Classification — Deep Learning Project

**Universidade do Minho | Aprendizagem Profunda | 2025/2026**

Automatic classification of ERCP (Endoscopic Retrograde Cholangiopancreatography) fluoroscopic images into 4 diagnostic categories using Deep Learning, with interpretability via Grad-CAM.

---

## Classes

| Class           | Description                    |
| --------------- | ------------------------------ |
| `Biliary_Leaks` | Bile leaks / Fugas de bílis    |
| `Lithiasis`     | Biliary stones / Cálculos      |
| `Stricture`     | Biliary strictures / Estenoses |
| `Normal`        | Normal findings                |

## Baseline to beat

**F1 macro = 0.738** (from the MIQR-CC paper)

## Our best result

**F1 macro = 0.7483** (E6 — stacking ensemble over E2 + E4 across 5 seeds)

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

# 4. Train a single experiment (e.g. our champion CNN, ConvNeXt-tiny + Focal, seed 42)
python scripts/train_experiment.py --model convnext_tiny --loss focal --tag E4 --seed 42

# 5. Evaluate any checkpoint
python scripts/evaluate_model.py \
    --checkpoint models/convnext_tiny_focal_E4_seed42.pth \
    --model convnext_tiny

# 6. Mixed-architecture ensemble (E6 — final reported result, stacking by default)
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
    --tag E6_ensemble_mixed

# 6b. Plain averaging variant (ablation): add --combine avg
python scripts/eval_ensemble.py --combine avg --checkpoints ... --tag E6_avg

# 7. Grad-CAM
python scripts/generate_gradcam.py --checkpoint models/convnext_tiny_focal_E4_seed42.pth

# 8. Run the full experiment suite (E0 → E6 end-to-end, ~10–12 h on a single GPU)
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
│   │   ├── build_model.py       # Model factory (ResNet/EfficientNet/DenseNet/ConvNeXt/DeiT)
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
│   ├── train_experiment.py      # Any model + hyperparams (supports --seed)
│   ├── evaluate_model.py        # Evaluate any checkpoint
│   ├── eval_ensemble.py         # Multi-ckpt + mixed-arch ensemble + TTA
│   ├── generate_gradcam.py      # Grad-CAM for any checkpoint
│   └── run_all_experiments.py   # Run E0–E6 end-to-end
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

## Experiments

All experiments use the same training pipeline (320 px input, 2-stage fine-tuning
with head-only warmup, RandAugment + RandomErasing, cosine LR with linear warmup,
mixed precision on GPU).

| ID             | Model                                                                          | CLAHE | Loss        | Sampler | Seeds | Test F1 macro    |
| -------------- | ------------------------------------------------------------------------------ | ----- | ----------- | ------- | ----- | ---------------- |
| E0             | ResNet18                                                                       | Yes   | CE weighted | No      | 1     | 0.5442           |
| E1             | EfficientNet-B0                                                                | Yes   | CE weighted | No      | 1     | 0.5470           |
| E2             | DenseNet121                                                                    | No    | Focal       | No      | 5     | 0.5421 (seed 42) |
| E3             | EfficientNet-B0                                                                | No    | CE weighted | Yes     | 1     | 0.6130           |
| E4             | ConvNeXt-tiny                                                                  | No    | Focal       | No      | 5     | 0.6730 (seed 42) |
| E5             | DeiT-tiny (ViT)                                                                | No    | Focal       | No      | 1     | 0.5447           |
| **E6**         | **Ensemble of E2 + E4 across 5 seeds (10 ckpts), Logistic Regression stacker** |       |             |         |       | **0.7483**       |
| E6′ (ablation) | Same checkpoints, plain logit averaging                                        |       |             |         |       | 0.7306           |

E2 and E4 are the ensemble feeders — they are trained with 5 seeds
(`42, 123, 777, 7, 2024`). The default E6 strategy fits a Logistic Regression
meta-learner (`C=0.01`, `class_weight=balanced`) on the validation set's
per-checkpoint softmax probabilities and applies it to the test set. The
single-seed column reports the seed-42 run for direct comparability with
E0/E1/E3/E5.

**Baseline (paper):** 0.738. **Best result here:** **0.7483 (E6, +0.010)**.

---

## Supported Models

- `resnet18`
- `resnet50`
- `efficientnet_b0`
- `efficientnet_b3`
- `densenet121` — ensemble feeder (E2)
- `mobilenet_v3`
- `deit_tiny` — Vision Transformer (E5)
- `convnext_tiny` — ensemble feeder, best single-model (E4)

## Dataset

- **Paper:** https://doi.org/10.6084/m9.figshare.31079236
- **Download:** https://figshare.com/ndownloader/files/61063177
- **GitHub reference:** https://github.com/monicaccmartins/MIQR-CC-Dataset
- **Total:** 1,602 patients | 19,317 processed images | 5,519 labelled
