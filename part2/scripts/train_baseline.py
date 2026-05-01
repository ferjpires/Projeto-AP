"""
Train the ResNet18 baseline model.

Usage:
    python scripts/train_baseline.py [--config config.yaml]
"""
import sys
import argparse
import copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.paths import load_config, ensure_dirs, get_project_root
from src.utils.seed import set_seed
from src.utils.plots import plot_learning_curves
from src.data.dataset import build_dataloaders
from src.models.build_model import build_model, count_parameters
from src.models.losses import build_loss
from src.training.train import train_model
from src.training.evaluate import evaluate_on_test, load_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    root = get_project_root()
    cfg = load_config(root / args.config)

    # Override model to resnet18 for baseline
    cfg = copy.deepcopy(cfg)
    cfg["model"]["name"] = "resnet18"
    cfg["training"]["num_epochs"] = min(cfg["training"]["num_epochs"], 25)

    set_seed(cfg["seed"])
    ensure_dirs(cfg)

    print("=" * 55)
    print("  BASELINE: ResNet18")
    print("=" * 55)

    # Data
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = \
        build_dataloaders(cfg)

    print(f"\n  Dataset sizes:")
    print(f"    Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")
    print(f"  Train class counts: {train_ds.get_class_counts()}")

    # Model
    model = build_model("resnet18", num_classes=cfg["model"]["num_classes"],
                        pretrained=cfg["model"]["pretrained"])
    params = count_parameters(model)
    print(f"\n  Parameters: {params['total']:,} total, {params['trainable']:,} trainable")

    # Loss with class weights
    import torch
    class_weights = train_ds.get_class_weights()
    device_str = cfg["training"].get("device", "cpu")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    criterion = build_loss(cfg, class_weights, device)
    print(f"  Class weights: {class_weights.tolist()}")

    # Train
    exp_dir = root / cfg["outputs"]["experiments_dir"] / "baseline_resnet18"
    history, best_ckpt = train_model(
        model, train_loader, val_loader, criterion, cfg,
        experiment_dir=exp_dir, model_name="resnet18_baseline"
    )

    # Learning curves
    plot_learning_curves(
        history["train_loss"], history["val_loss"],
        history["train_f1"], history["val_f1"],
        save_path=root / cfg["outputs"]["figures_dir"] / "resnet18_baseline_curves.png",
        title="ResNet18 Baseline"
    )

    # Load best and evaluate on test
    model = load_checkpoint(str(best_ckpt), model, device)
    evaluate_on_test(model, test_loader, cfg,
                     save_dir=root / cfg["outputs"]["tables_dir"],
                     model_name="resnet18_baseline")

    # Copy best model
    import shutil
    models_dir = root / cfg["outputs"]["models_dir"]
    models_dir.mkdir(exist_ok=True)
    shutil.copy(best_ckpt, models_dir / "baseline_resnet18.pth")
    print(f"\n  Best model saved to {models_dir / 'baseline_resnet18.pth'}")


if __name__ == "__main__":
    main()
