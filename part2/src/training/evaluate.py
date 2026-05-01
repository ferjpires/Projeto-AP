"""
Evaluation utilities: run inference on a DataLoader and collect predictions.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path


@torch.no_grad()
def run_inference(model: nn.Module, loader: DataLoader,
                  device: torch.device) -> tuple:
    """
    Run model on all batches in loader.

    Returns:
        y_true:  (N,) int array of ground-truth labels
        y_pred:  (N,) int array of predicted labels
        y_prob:  (N, C) float array of softmax probabilities
    """
    model.eval()
    all_true, all_pred, all_prob = [], [], []

    for imgs, labels in tqdm(loader, desc="  Inference", leave=False):
        imgs = imgs.to(device)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        all_true.extend(labels.numpy())
        all_pred.extend(preds)
        all_prob.append(probs)

    return (
        np.array(all_true),
        np.array(all_pred),
        np.vstack(all_prob),
    )


def load_checkpoint(checkpoint_path: str, model: nn.Module,
                    device: torch.device) -> nn.Module:
    """Load model weights from a checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model = model.to(device)
    epoch = ckpt.get("epoch", "?")
    f1 = ckpt.get("val_f1_macro", "?")
    print(f"  Loaded checkpoint from epoch {epoch}  (val F1={f1})")
    return model


def evaluate_on_test(model: nn.Module, test_loader: DataLoader,
                     config: dict, save_dir: Path,
                     model_name: str = "model") -> dict:
    """
    Full test-set evaluation: inference → metrics → plots → CSV.
    """
    from src.training.metrics import (compute_metrics, print_metrics,
                                      save_metrics_csv, save_classification_report)
    from src.utils.plots import plot_confusion_matrix, plot_roc_curves

    device_str = config["training"].get("device", "cpu")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    class_names = config["data"]["class_names"]
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    y_true, y_pred, y_prob = run_inference(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred, y_prob, class_names)

    print_metrics(metrics, class_names, model_name)

    # Confusion matrix
    cm_path = Path(config["outputs"]["confusion_matrices_dir"]) / f"{model_name}_cm.png"
    cm_path.parent.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(metrics["confusion_matrix"], class_names,
                          cm_path, title=f"Confusion Matrix — {model_name}")

    # ROC curves
    roc_path = Path(config["outputs"]["roc_dir"]) / f"{model_name}_roc.png"
    roc_path.parent.mkdir(parents=True, exist_ok=True)
    if not np.isnan(y_prob).any():
        plot_roc_curves(y_true, y_prob, class_names, roc_path)

    # Classification report
    save_classification_report(
        y_true, y_pred, class_names,
        save_dir / f"{model_name}_classification_report.txt"
    )

    # Metrics CSV
    save_metrics_csv(
        metrics,
        save_dir / f"{model_name}_metrics.csv",
        extra_info={"model": model_name}
    )

    return metrics
