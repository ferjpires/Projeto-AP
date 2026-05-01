"""
Exploratory Data Analysis utilities for the MIQR-CC dataset.
"""
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm


CLASS_NAMES = ["Biliary_Leaks", "Lithiasis", "Normal", "Stricture"]


def count_images(split_dir: str, class_names: List[str] = None) -> Dict[str, int]:
    class_names = class_names or CLASS_NAMES
    root = Path(split_dir)
    counts = {}
    ext = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    for cls in class_names:
        d = root / cls
        if d.exists():
            counts[cls] = sum(1 for f in d.iterdir() if f.suffix.lower() in ext)
        else:
            counts[cls] = 0
    return counts


def get_image_sizes(split_dir: str, class_names: List[str] = None,
                    max_per_class: int = 100) -> List[tuple]:
    class_names = class_names or CLASS_NAMES
    root = Path(split_dir)
    ext = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    sizes = []
    for cls in class_names:
        d = root / cls
        if not d.exists():
            continue
        files = [f for f in d.iterdir() if f.suffix.lower() in ext][:max_per_class]
        for f in files:
            try:
                w, h = Image.open(f).size
                sizes.append((w, h))
            except Exception:
                pass
    return sizes


def get_sample_paths(split_dir: str, class_names: List[str] = None,
                     n: int = 4) -> Dict[str, List[Path]]:
    class_names = class_names or CLASS_NAMES
    root = Path(split_dir)
    ext = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    result = {}
    for cls in class_names:
        d = root / cls
        if not d.exists():
            result[cls] = []
            continue
        files = sorted([f for f in d.iterdir() if f.suffix.lower() in ext])
        result[cls] = files[:n]
    return result


def run_eda(config: dict, output_dir: str = None) -> None:
    """Run full EDA and save all figures."""
    from src.utils.plots import plot_class_distribution, plot_sample_images

    out = Path(output_dir or config["outputs"]["figures_dir"])
    out.mkdir(parents=True, exist_ok=True)
    class_names = config["data"]["class_names"]

    for split in ["train", "val", "test"]:
        split_dir = config["data"][f"{split}_dir"]
        counts = count_images(split_dir, class_names)
        total = sum(counts.values())
        print(f"\n=== {split.upper()} ===")
        for cls, n in counts.items():
            pct = 100 * n / total if total > 0 else 0
            print(f"  {cls}: {n} ({pct:.1f}%)")
        print(f"  Total: {total}")
        plot_class_distribution(counts, split.capitalize(),
                                out / f"class_distribution_{split}.png")

    # Image sizes (from train)
    sizes = get_image_sizes(config["data"]["train_dir"], class_names)
    if sizes:
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(widths, bins=30, color="#3498db", edgecolor="black")
        axes[0].set_title("Image Width Distribution (train sample)")
        axes[0].set_xlabel("Width (px)")
        axes[1].hist(heights, bins=30, color="#e74c3c", edgecolor="black")
        axes[1].set_title("Image Height Distribution (train sample)")
        axes[1].set_xlabel("Height (px)")
        plt.tight_layout()
        plt.savefig(out / "image_size_distribution.png", dpi=150)
        plt.close()
        print(f"\nImage sizes — W: {np.min(widths)}–{np.max(widths)} px,"
              f"  H: {np.min(heights)}–{np.max(heights)} px")

    # Sample images
    train_samples = get_sample_paths(config["data"]["train_dir"], class_names, n=4)
    has_images = any(len(v) > 0 for v in train_samples.values())
    if has_images:
        plot_sample_images(train_samples, out / "sample_images_per_class.png", n_per_class=4)
        print(f"\nEDA figures saved to {out}/")
    else:
        print("\n[INFO] No training images found — skipping sample image plot.")
        print("       Place your dataset in data/processed/train/<class>/ first.")
