import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
import itertools


CLASS_NAMES = ["Biliary_Leaks", "Lithiasis", "Normal", "Stricture"]


def plot_class_distribution(counts: dict, split: str, save_path: Path) -> None:
    """Bar chart of class distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))
    classes = list(counts.keys())
    values = list(counts.values())
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    bars = ax.bar(classes, values, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_title(f"Class Distribution — {split}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Number of Images", fontsize=12)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(val), ha="center", va="bottom", fontsize=11)
    ax.set_ylim(0, max(values) * 1.15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_learning_curves(train_losses, val_losses, train_f1s, val_f1s,
                         save_path: Path, title: str = "Learning Curves") -> None:
    """Plot training/validation loss and F1 macro curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, "b-o", markersize=4, label="Train Loss")
    ax1.plot(epochs, val_losses, "r-o", markersize=4, label="Val Loss")
    ax1.set_title(f"{title} — Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_f1s, "b-o", markersize=4, label="Train F1 macro")
    ax2.plot(epochs, val_f1s, "r-o", markersize=4, label="Val F1 macro")
    ax2.axhline(y=0.738, color="green", linestyle="--", linewidth=1.5, label="Baseline 0.738")
    ax2.set_title(f"{title} — F1 Macro")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Macro")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: list,
                          save_path: Path, title: str = "Confusion Matrix") -> None:
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_curves(y_true: np.ndarray, y_prob: np.ndarray,
                    class_names: list, save_path: Path) -> None:
    """Plot one-vs-rest ROC curves for each class."""
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    aucs = []
    for i, (cls, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    macro_auc = np.mean(aucs)
    ax.set_title(f"ROC Curves (Macro AUC = {macro_auc:.3f})", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return macro_auc


def plot_sample_images(image_paths_by_class: dict, save_path: Path,
                       n_per_class: int = 4) -> None:
    """Show sample images for each class in a grid."""
    n_classes = len(image_paths_by_class)
    fig, axes = plt.subplots(n_classes, n_per_class, figsize=(n_per_class * 3, n_classes * 3))

    for row, (cls, paths) in enumerate(image_paths_by_class.items()):
        for col in range(n_per_class):
            ax = axes[row][col] if n_classes > 1 else axes[col]
            if col < len(paths):
                from PIL import Image
                img = Image.open(paths[col]).convert("RGB")
                ax.imshow(img, cmap="gray")
                if col == 0:
                    ax.set_ylabel(cls, fontsize=10, fontweight="bold", rotation=90)
            ax.axis("off")

    plt.suptitle("Sample Images per Class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
