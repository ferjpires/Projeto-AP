"""
Evaluate a saved checkpoint on the test set.

Usage:
    python scripts/evaluate_model.py --checkpoint models/best_model.pth
    python scripts/evaluate_model.py --checkpoint models/best_model.pth --split val
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.utils.paths import load_config, get_project_root
from src.utils.seed import set_seed
from src.data.dataset import build_dataloaders
from src.models.build_model import build_model
from src.training.evaluate import evaluate_on_test, load_checkpoint, run_inference
from src.training.metrics import compute_metrics, print_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Path to .pth checkpoint file")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--model", default=None,
                   help="Model name override (if not stored in checkpoint)")
    return p.parse_args()


def main():
    args = parse_args()
    root = get_project_root()
    cfg = load_config(root / args.config)
    set_seed(cfg["seed"])

    device_str = cfg["training"].get("device", "cpu")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # Load checkpoint to get model name
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_name = args.model or ckpt.get("model_name", cfg["model"]["name"])
    # Strip experiment suffix for build_model
    base_name = model_name.split("_weighted")[0].split("_focal")[0].split("_clahe")[0]
    # Try to match to a known model
    known = ["resnet18", "resnet50", "efficientnet_b0", "efficientnet_b3",
             "densenet121", "mobilenet_v3", "deit_tiny"]
    model_key = next((m for m in known if m in base_name.lower()), cfg["model"]["name"])

    print(f"\n  Model: {model_name} (architecture: {model_key})")

    model = build_model(model_key, num_classes=cfg["model"]["num_classes"],
                        pretrained=False)
    model = load_checkpoint(args.checkpoint, model, device)

    train_l, val_l, test_l, train_ds, val_ds, test_ds = build_dataloaders(cfg)
    loader_map = {"train": train_l, "val": val_l, "test": test_l}
    loader = loader_map[args.split]

    print(f"  Evaluating on: {args.split} set")
    metrics = evaluate_on_test(
        model, loader, cfg,
        save_dir=root / cfg["outputs"]["tables_dir"],
        model_name=f"{model_name}_{args.split}"
    )


if __name__ == "__main__":
    main()
