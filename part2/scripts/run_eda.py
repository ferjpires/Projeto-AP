"""
Run Exploratory Data Analysis on the MIQR-CC dataset.

Usage:
    python scripts/run_eda.py [--config config.yaml]
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.paths import load_config, ensure_dirs, get_project_root
from src.utils.seed import set_seed
from src.data.eda import run_eda


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    root = get_project_root()
    config = load_config(root / args.config)
    set_seed(config["seed"])
    ensure_dirs(config)
    print("\n=== Running EDA ===")
    run_eda(config, output_dir=config["outputs"]["figures_dir"])
    print("\nEDA complete.")


if __name__ == "__main__":
    main()
