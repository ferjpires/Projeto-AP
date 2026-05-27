"""
Multi-model / multi-seed ensemble evaluation with optional horizontal-flip TTA.

Each checkpoint is given as ``arch:path`` so a single ensemble may mix
architectures (e.g. DenseNet + ConvNeXt). For each checkpoint, logits are
collected on the test set with — and optionally without — a horizontal flip.

Two combination strategies are supported:

  * ``--combine stack`` (default): a Logistic Regression meta-learner is fit
    on the validation set's per-checkpoint softmax probabilities (one feature
    per checkpoint and class, so 4·K features). The fitted classifier then
    produces the test predictions. This was the project's best strategy.

  * ``--combine avg``: simple arithmetic mean of logits across all
    (checkpoint × TTA) views. Reported as ablation in the paper.

Usage:
    # Final reported result (stacking, no TTA)
    python scripts/eval_ensemble.py \\
        --checkpoints \\
            densenet121:models/densenet121_focal_E2_seed42.pth \\
            densenet121:models/densenet121_focal_E2_seed123.pth \\
            convnext_tiny:models/convnext_tiny_focal_E4_seed42.pth \\
            convnext_tiny:models/convnext_tiny_focal_E4_seed123.pth \\
        --tag E6_ensemble_mixed_stack

    # Plain averaging
    python scripts/eval_ensemble.py --combine avg --checkpoints ... --tag E6_avg
"""
import sys
import argparse
import multiprocessing as mp
from pathlib import Path

try:
    mp.set_start_method("fork", force=True)
except RuntimeError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from src.utils.paths import load_config, get_project_root
from src.data.dataset import ERCPDataset
from src.data.transforms import get_transforms
from src.models.build_model import build_model
from src.training.metrics import (compute_metrics, print_metrics,
                                  save_metrics_csv, save_classification_report)
from src.utils.plots import plot_confusion_matrix, plot_roc_curves


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="Checkpoints to ensemble, each as 'arch:path'.")
    p.add_argument("--tta", choices=["none", "hflip"], default="none",
                   help="TTA mode. 'none' = identity only; "
                        "'hflip' = identity + horizontal flip.")
    p.add_argument("--combine", choices=["stack", "avg"], default="stack",
                   help="How to combine per-ckpt predictions. "
                        "'stack' fits a Logistic Regression on val-set "
                        "probabilities (default; best result). "
                        "'avg' averages logits across all views.")
    p.add_argument("--stacker-C", type=float, default=0.01,
                   help="L2 regularization strength for the stacker "
                        "(only used when --combine stack). Smaller = more "
                        "regularization. 0.01 was best on this dataset.")
    p.add_argument("--tag", default=None,
                   help="Name used for output files (defaults to 'ensemble<N>').")
    return p.parse_args()


def parse_ckpt_spec(spec: str, default_arch: str) -> tuple[str, str]:
    if ":" in spec and not spec.startswith("/"):
        arch, _, path = spec.partition(":")
        return arch.strip(), path.strip()
    return default_arch, spec


@torch.no_grad()
def collect_logits(model, loader, device, hflip: bool) -> torch.Tensor:
    model.eval()
    out = []
    for imgs, _ in loader:
        imgs = imgs.to(device)
        if hflip:
            imgs = torch.flip(imgs, dims=[3])
        out.append(model(imgs).cpu())
    return torch.cat(out, 0)


def _build_loader(cfg, split_dir, img_size, use_clahe, class_names,
                  batch, n_workers):
    ds = ERCPDataset(split_dir,
                     transform=get_transforms(img_size, "test", use_clahe),
                     class_names=class_names)
    loader = DataLoader(ds, batch_size=batch, shuffle=False,
                        num_workers=n_workers,
                        pin_memory=torch.cuda.is_available())
    y = np.array([y for _, y in ds.samples])
    return ds, loader, y


def main():
    args = parse_args()
    root = get_project_root()
    cfg = load_config(root / args.config)
    default_arch = cfg["model"]["name"]

    device_str = cfg["training"].get("device", "cpu")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    img_size = cfg["training"]["image_size"]
    batch = cfg["training"]["batch_size"]
    n_workers = cfg["training"].get("num_workers", 2)
    use_clahe = cfg["augmentation"].get("use_clahe", False)
    class_names = cfg["data"]["class_names"]

    _, test_loader, y_true = _build_loader(
        cfg, cfg["data"]["test_dir"], img_size, use_clahe, class_names,
        batch, n_workers)

    if args.combine == "stack":
        _, val_loader, y_val = _build_loader(
            cfg, cfg["data"]["val_dir"], img_size, use_clahe, class_names,
            batch, n_workers)

    ckpts = [parse_ckpt_spec(s, default_arch) for s in args.checkpoints]
    tta_modes = [False] if args.tta == "none" else [False, True]
    print(f"\nEnsembling {len(ckpts)} ckpts × {len(tta_modes)} TTA view(s) "
          f"= {len(ckpts) * len(tta_modes)} forward passes per split "
          f"(combine={args.combine})")

    test_logits_views = []     # list of (N_test, C) tensors, one per (ckpt × TTA)
    val_probs_per_ckpt = []    # list of (N_val, C), one per ckpt (TTA-averaged probs)
    test_probs_per_ckpt = []   # list of (N_test, C), one per ckpt (TTA-averaged probs)

    for arch, path in ckpts:
        print(f"  [{arch}] {path}")
        model = build_model(arch, num_classes=cfg["model"]["num_classes"],
                            pretrained=False).to(device)
        ckpt = torch.load(path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)

        # Test pass (collect per-view logits AND TTA-averaged probs per ckpt)
        ckpt_test_views = []
        for hflip in tta_modes:
            L = collect_logits(model, test_loader, device, hflip)
            test_logits_views.append(L)
            ckpt_test_views.append(L)
        ckpt_test_logits = torch.stack(ckpt_test_views, 0).mean(0)
        test_probs_per_ckpt.append(torch.softmax(ckpt_test_logits, dim=1).numpy())

        # Val pass only for stacking
        if args.combine == "stack":
            ckpt_val_views = []
            for hflip in tta_modes:
                ckpt_val_views.append(collect_logits(model, val_loader, device, hflip))
            ckpt_val_logits = torch.stack(ckpt_val_views, 0).mean(0)
            val_probs_per_ckpt.append(torch.softmax(ckpt_val_logits, dim=1).numpy())

    K = len(ckpts)
    C = cfg["model"]["num_classes"]

    if args.combine == "stack":
        # Feature vectors: concatenate softmax probabilities from each ckpt.
        val_probs  = np.stack(val_probs_per_ckpt,  axis=0)   # (K, N_val,  C)
        test_probs = np.stack(test_probs_per_ckpt, axis=0)   # (K, N_test, C)
        N_val  = val_probs.shape[1]
        N_test = test_probs.shape[1]
        X_val  = val_probs.transpose(1, 0, 2).reshape(N_val,  K * C)
        X_test = test_probs.transpose(1, 0, 2).reshape(N_test, K * C)

        stacker = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=args.stacker_C, max_iter=5000,
                               class_weight="balanced", solver="lbfgs"),
        )
        stacker.fit(X_val, y_val)
        val_pred = stacker.predict(X_val)
        from sklearn.metrics import f1_score
        print(f"  Stacker fitted (C={args.stacker_C}): "
              f"val F1 macro={f1_score(y_val, val_pred, average='macro', zero_division=0):.4f}")
        y_prob = stacker.predict_proba(X_test)
        y_pred = y_prob.argmax(axis=1)
    else:
        ens_logits = torch.stack(test_logits_views, 0).mean(0)
        y_prob = torch.softmax(ens_logits, dim=1).numpy()
        y_pred = y_prob.argmax(axis=1)

    metrics = compute_metrics(y_true, y_pred, y_prob, class_names)
    tag = args.tag or f"ensemble{len(ckpts)}_{args.combine}"
    print_metrics(metrics, class_names, tag)

    out_tables = root / cfg["outputs"]["tables_dir"]
    out_tables.mkdir(parents=True, exist_ok=True)
    save_classification_report(y_true, y_pred, class_names,
                               out_tables / f"{tag}_classification_report.txt")
    save_metrics_csv(metrics, out_tables / f"{tag}_metrics.csv",
                     extra_info={"model": tag,
                                 "n_checkpoints": len(ckpts),
                                 "tta": args.tta,
                                 "combine": args.combine,
                                 "archs": ",".join(sorted({a for a, _ in ckpts}))})

    cm_path = root / cfg["outputs"]["confusion_matrices_dir"] / f"{tag}_cm.png"
    cm_path.parent.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(metrics["confusion_matrix"], class_names,
                          cm_path, title=f"Confusion Matrix — {tag}")

    roc_path = root / cfg["outputs"]["roc_dir"] / f"{tag}_roc.png"
    roc_path.parent.mkdir(parents=True, exist_ok=True)
    if not np.isnan(y_prob).any():
        plot_roc_curves(y_true, y_prob, class_names, roc_path)

    print(f"\nEnsemble test F1 macro: {metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
