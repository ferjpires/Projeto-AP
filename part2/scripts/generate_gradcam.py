"""
Generate Grad-CAM visualizations for a trained model.

Usage:
    python scripts/generate_gradcam.py --checkpoint models/best_model.pth
    python scripts/generate_gradcam.py --checkpoint models/best_model.pth --n 5 --split test
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.utils.paths import load_config, ensure_dirs, get_project_root
from src.utils.seed import set_seed
from src.data.dataset import ERCPDataset
from src.data.transforms import get_transforms
from src.models.build_model import build_model
from src.training.evaluate import load_checkpoint
from src.explainability.gradcam import generate_gradcam_examples


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--n", type=int, default=3,
                   help="Number of examples per class")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--model", default=None,
                   help="Override model architecture name")
    return p.parse_args()


def main():
    args = parse_args()
    root = get_project_root()
    cfg = load_config(root / args.config)
    set_seed(cfg["seed"])
    ensure_dirs(cfg)

    device_str = cfg["training"].get("device", "cpu")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_name = args.model or ckpt.get("model_name", cfg["model"]["name"])
    known = ["resnet18", "resnet50", "efficientnet_b0", "efficientnet_b3",
             "densenet121", "mobilenet_v3", "deit_tiny"]
    model_key = next((m for m in known if m in model_name.lower()), cfg["model"]["name"])

    print(f"\n  Model: {model_name} (arch: {model_key})")
    print(f"  Split: {args.split}  |  N per class: {args.n}")

    model = build_model(model_key, num_classes=cfg["model"]["num_classes"],
                        pretrained=False)
    model = load_checkpoint(args.checkpoint, model, device)

    split_dir = cfg["data"][f"{args.split}_dir"]
    aug_cfg = cfg.get("augmentation", {})
    use_clahe = aug_cfg.get("use_clahe", False)
    dataset = ERCPDataset(
        split_dir,
        transform=get_transforms(cfg["training"]["image_size"], "test", use_clahe),
        class_names=cfg["data"]["class_names"],
    )

    print(f"  Dataset: {len(dataset)} images")
    generate_gradcam_examples(model, dataset, model_key, cfg,
                              n_per_class=args.n)

    print(f"\n  Grad-CAM images saved to {cfg['outputs']['gradcam_dir']}/")


if __name__ == "__main__":
    main()
