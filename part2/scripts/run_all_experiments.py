"""
Run all planned experiments sequentially and produce a comparison table.

Usage:
    python scripts/run_all_experiments.py [--config config.yaml] [--quick]

--quick flag reduces epochs to 5 for a fast smoke-test of the pipeline.
"""
import sys
import subprocess
import argparse
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EXPERIMENTS = [
    # (model, clahe, loss, sampler, tag)
    ("resnet18",        False, "weighted_cross_entropy", False, "E0"),
    ("resnet18",        True,  "weighted_cross_entropy", False, "E1"),
    ("efficientnet_b0", False, "weighted_cross_entropy", False, "E2"),
    ("efficientnet_b0", True,  "weighted_cross_entropy", False, "E3"),
    ("densenet121",     False, "focal",                  False, "E4"),
    ("efficientnet_b0", False, "weighted_cross_entropy", True,  "E5"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--quick", action="store_true",
                   help="Run only 5 epochs per model (smoke test)")
    return p.parse_args()


def run_experiment(model, clahe, loss, sampler, tag, config, quick):
    cmd = [sys.executable, "scripts/train_experiment.py",
           "--config", config,
           "--model", model,
           "--loss", loss,
           "--tag", tag]
    if clahe:
        cmd.append("--clahe")
    if sampler:
        cmd.append("--sampler")
    if quick:
        cmd += ["--epochs", "5"]
    print(f"\n{'#'*60}")
    print(f"  Running: {' '.join(cmd)}")
    print(f"{'#'*60}")
    subprocess.run(cmd, check=True)


def aggregate_results(config_path):
    from src.utils.paths import load_config, get_project_root
    cfg = load_config(config_path)
    root = get_project_root()
    tables_dir = root / cfg["outputs"]["tables_dir"]

    dfs = []
    for csv_file in tables_dir.glob("*_metrics.csv"):
        if "test" not in csv_file.name and "val" not in csv_file.name:
            try:
                df = pd.read_csv(csv_file)
                dfs.append(df)
            except Exception:
                pass

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        out = tables_dir / "all_experiments_summary.csv"
        combined.to_csv(out, index=False)
        print(f"\nSummary table saved to {out}")
        cols = ["model", "f1_macro", "auc_macro", "accuracy",
                "precision_macro", "recall_macro"]
        cols = [c for c in cols if c in combined.columns]
        print("\n" + combined[cols].sort_values("f1_macro", ascending=False).to_string())


def main():
    args = parse_args()
    config = args.config

    for model, clahe, loss, sampler, tag in EXPERIMENTS:
        try:
            run_experiment(model, clahe, loss, sampler, tag, config, args.quick)
        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] Experiment {tag} failed: {e}")
            continue

    print("\n\n=== All experiments finished ===")
    aggregate_results(config)


if __name__ == "__main__":
    main()
