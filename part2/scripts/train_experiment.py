"""
Train an experiment with any supported model.

Usage:
    python scripts/train_experiment.py --model efficientnet_b0 [--config config.yaml]
    python scripts/train_experiment.py --model densenet121 --clahe --loss focal
    python scripts/train_experiment.py --model efficientnet_b3 --sampler
"""
import sys
import argparse
import copy
import shutil
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
    p.add_argument("--model", default=None,
                   help="Override model name (e.g. efficientnet_b0, densenet121)")
    p.add_argument("--clahe", action="store_true",
                   help="Enable CLAHE preprocessing")
    p.add_argument("--loss", default=None,
                   choices=["weighted_cross_entropy", "focal"],
                   help="Override loss function")
    p.add_argument("--sampler", action="store_true",
                   help="Use WeightedRandomSampler instead of class weights in loss")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--tag", default=None,
                   help="Extra tag to append to experiment name")
    return p.parse_args()


def main():
    args = parse_args()
    root = get_project_root()
    cfg = load_config(root / args.config)
    cfg = copy.deepcopy(cfg)

    # Apply overrides
    if args.model:
        cfg["model"]["name"] = args.model
    if args.clahe:
        cfg["augmentation"]["use_clahe"] = True
    if args.loss:
        cfg["loss"]["name"] = args.loss
    if args.sampler:
        cfg.setdefault("sampling", {})["use_weighted_sampler"] = True
    if args.epochs:
        cfg["training"]["num_epochs"] = args.epochs
    if args.lr:
        cfg["training"]["learning_rate"] = args.lr

    model_name = cfg["model"]["name"]
    clahe_tag = "_clahe" if cfg["augmentation"].get("use_clahe") else ""
    loss_tag = f"_{cfg['loss']['name']}"
    sampler_tag = "_sampler" if cfg.get("sampling", {}).get("use_weighted_sampler") else ""
    extra_tag = f"_{args.tag}" if args.tag else ""
    exp_name = f"{model_name}{clahe_tag}{loss_tag}{sampler_tag}{extra_tag}"

    print("=" * 55)
    print(f"  Experiment: {exp_name}")
    print(f"  Model: {model_name} | CLAHE: {cfg['augmentation'].get('use_clahe')}")
    print(f"  Loss: {cfg['loss']['name']} | Sampler: {cfg.get('sampling', {}).get('use_weighted_sampler')}")
    print("=" * 55)

    set_seed(cfg["seed"])
    ensure_dirs(cfg)

    import torch
    device_str = cfg["training"].get("device", "cpu")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = \
        build_dataloaders(cfg)
    print(f"\n  Dataset — Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")
    print(f"  Class counts (train): {train_ds.get_class_counts()}")

    # Model
    model = build_model(model_name, num_classes=cfg["model"]["num_classes"],
                        pretrained=cfg["model"]["pretrained"])
    params = count_parameters(model)
    print(f"\n  Parameters: {params['total']:,} total, {params['trainable']:,} trainable")

    # Loss
    class_weights = train_ds.get_class_weights() if not cfg.get("sampling", {}).get(
        "use_weighted_sampler") else None
    criterion = build_loss(cfg, class_weights, device)

    # Train
    exp_dir = root / cfg["outputs"]["experiments_dir"] / exp_name
    history, best_ckpt = train_model(
        model, train_loader, val_loader, criterion, cfg,
        experiment_dir=exp_dir, model_name=exp_name
    )

    # Plots
    plot_learning_curves(
        history["train_loss"], history["val_loss"],
        history["train_f1"], history["val_f1"],
        save_path=root / cfg["outputs"]["figures_dir"] / f"{exp_name}_curves.png",
        title=exp_name
    )

    # Evaluate on test
    model = load_checkpoint(str(best_ckpt), model, device)
    metrics = evaluate_on_test(
        model, test_loader, cfg,
        save_dir=root / cfg["outputs"]["tables_dir"],
        model_name=exp_name
    )

    # Save best model
    models_dir = root / cfg["outputs"]["models_dir"]
    models_dir.mkdir(exist_ok=True)
    shutil.copy(best_ckpt, models_dir / f"{exp_name}.pth")

    f1 = metrics.get("f1_macro", 0)
    print(f"\n  Final test F1 macro: {f1:.4f}")
    if f1 >= 0.738:
        print("  ✓ BEATS baseline (0.738)")
    else:
        print("  Baseline not yet beaten — try more epochs or different model.")


if __name__ == "__main__":
    main()
