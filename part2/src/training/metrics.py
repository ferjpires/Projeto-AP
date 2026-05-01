"""
Evaluation metrics for ERCP multi-class classification.
Main metric: F1 macro (as required by the assignment).
"""
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, accuracy_score, roc_auc_score,
)
from sklearn.preprocessing import label_binarize
import pandas as pd
from pathlib import Path


CLASS_NAMES = ["Biliary_Leaks", "Lithiasis", "Normal", "Stricture"]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray = None,
                    class_names: list = None) -> dict:
    """
    Compute the full set of evaluation metrics.

    Returns a dict with scalar values for the summary table and
    per-class breakdown for the report.
    """
    class_names = class_names or CLASS_NAMES

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

    # Per-class F1
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    for cls, val in zip(class_names, f1_per_class):
        metrics[f"f1_{cls}"] = val

    # AUC-ROC (needs probabilities)
    if y_prob is not None:
        try:
            n_cls = len(class_names)
            y_bin = label_binarize(y_true, classes=list(range(n_cls)))
            auc_macro = roc_auc_score(y_bin, y_prob, average="macro",
                                      multi_class="ovr")
            metrics["auc_macro"] = auc_macro
        except Exception as e:
            metrics["auc_macro"] = float("nan")
            print(f"  [WARN] AUC computation failed: {e}")
    else:
        metrics["auc_macro"] = float("nan")

    # Confusion matrix
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)

    return metrics


def print_metrics(metrics: dict, class_names: list = None,
                  model_name: str = "") -> None:
    class_names = class_names or CLASS_NAMES
    sep = "=" * 55
    print(f"\n{sep}")
    if model_name:
        print(f"  Model: {model_name}")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  F1 Macro:          {metrics['f1_macro']:.4f}  ← main metric")
    print(f"  F1 Weighted:       {metrics['f1_weighted']:.4f}")
    print(f"  Precision Macro:   {metrics['precision_macro']:.4f}")
    print(f"  Recall Macro:      {metrics['recall_macro']:.4f}")
    print(f"  AUC Macro:         {metrics['auc_macro']:.4f}")
    print(f"\n  Per-class F1:")
    for cls in class_names:
        key = f"f1_{cls}"
        print(f"    {cls:<20} {metrics.get(key, float('nan')):.4f}")
    baseline = 0.738
    diff = metrics["f1_macro"] - baseline
    symbol = "✓ BEATS" if diff >= 0 else "✗ BELOW"
    print(f"\n  Baseline F1 Macro: {baseline:.3f}")
    print(f"  {symbol} baseline by {abs(diff):.4f}")
    print(sep)


def save_metrics_csv(metrics: dict, save_path: Path,
                     extra_info: dict = None) -> None:
    """Save scalar metrics to a CSV row."""
    row = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
    if extra_info:
        row.update(extra_info)
    df = pd.DataFrame([row])
    df.to_csv(save_path, index=False)


def save_classification_report(y_true, y_pred, class_names, save_path: Path) -> None:
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   zero_division=0)
    with open(save_path, "w") as f:
        f.write(report)
    print(f"  Classification report saved to {save_path}")
