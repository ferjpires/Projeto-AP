"""
Run every experiment defined for this project, in order, and finish by
evaluating the mixed-architecture ensemble (E6).

Layout:
    E0 — ResNet18 + CLAHE + weighted cross-entropy
    E1 — EfficientNet-B0 + CLAHE + weighted cross-entropy
    E2 — DenseNet121 + focal  ◀ feeds the ensemble (5 seeds)
    E3 — EfficientNet-B0 + weighted cross-entropy + sampler
    E4 — ConvNeXt-tiny + focal  ◀ feeds the ensemble (5 seeds)
    E5 — DeiT-tiny + focal (Vision Transformer)
    E6 — Mixed-architecture ensemble of E2 + E4 across 5 seeds

Usage:
    python scripts/run_all_experiments.py [--config config.yaml] [--quick]

    --quick reduces every training run to 5 epochs (smoke test).
"""
import sys
import subprocess
import argparse
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# (tag, model, clahe, loss, sampler)
EXPERIMENTS = [
    ("E0", "resnet18",        True,  "weighted_cross_entropy", False),
    ("E1", "efficientnet_b0", True,  "weighted_cross_entropy", False),
    ("E2", "densenet121",     False, "focal",                  False),
    ("E3", "efficientnet_b0", False, "weighted_cross_entropy", True),
    ("E4", "convnext_tiny",   False, "focal",                  False),
    ("E5", "deit_tiny",       False, "focal",                  False),
]

# Tags whose checkpoints feed E6. Each is trained with every seed below
ENSEMBLE_FEEDER_TAGS = {"E2", "E4"}
ENSEMBLE_SEEDS = [42, 123, 777, 7, 2024]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--quick", action="store_true",
                   help="Run only 5 epochs per model (smoke test).")
    return p.parse_args()


def _exp_name(model, clahe, loss, sampler, tag, seed=None):
    """Mirror the naming convention used by train_experiment.py."""
    parts = [model]
    if clahe:
        parts.append("clahe")
    parts.append(loss)
    if sampler:
        parts.append("sampler")
    parts.append(tag)
    if seed is not None:
        parts.append(f"seed{seed}")
    return "_".join(parts)


def _train_cmd(model, clahe, loss, sampler, tag, seed, config, quick):
    cmd = [sys.executable, "scripts/train_experiment.py",
           "--config", config,
           "--model", model,
           "--loss", loss,
           "--tag", tag]
    if clahe:
        cmd.append("--clahe")
    if sampler:
        cmd.append("--sampler")
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if quick:
        cmd += ["--epochs", "5"]
    return cmd


def run_experiment(tag, model, clahe, loss, sampler, config, quick):
    """Run one E* experiment.

    If the tag feeds the ensemble, run all ENSEMBLE_SEEDS; otherwise use a
    single seed taken from config.yaml.
    """
    if tag in ENSEMBLE_FEEDER_TAGS:
        for seed in ENSEMBLE_SEEDS:
            cmd = _train_cmd(model, clahe, loss, sampler, tag, seed, config, quick)
            print(f"\n{'#'*60}\n  Running {tag} (seed={seed}): {' '.join(cmd)}\n{'#'*60}")
            subprocess.run(cmd, check=True)
    else:
        cmd = _train_cmd(model, clahe, loss, sampler, tag, None, config, quick)
        print(f"\n{'#'*60}\n  Running {tag}: {' '.join(cmd)}\n{'#'*60}")
        subprocess.run(cmd, check=True)


def run_ensemble(config, models_dir, tag="E6_ensemble_mixed"):
    """Average logits from every ensemble feeder checkpoint."""
    ckpts = []
    for exp_tag, model, clahe, loss, sampler in EXPERIMENTS:
        if exp_tag not in ENSEMBLE_FEEDER_TAGS:
            continue
        for seed in ENSEMBLE_SEEDS:
            name = _exp_name(model, clahe, loss, sampler, exp_tag, seed)
            path = models_dir / f"{name}.pth"
            if not path.exists():
                print(f"  [WARN] missing checkpoint, skipping: {path}")
                continue
            ckpts.append(f"{model}:{path}")
    if not ckpts:
        print("  [ERROR] no ensemble feeder checkpoints found; skipping E6.")
        return

    cmd = [sys.executable, "scripts/eval_ensemble.py",
           "--config", config,
           "--checkpoints", *ckpts,
           "--tta", "none",
           "--tag", tag]
    print(f"\n{'#'*60}\n  Running E6 ensemble: {len(ckpts)} ckpts\n{'#'*60}")
    subprocess.run(cmd, check=True)


def aggregate_results(config_path):
    from src.utils.paths import load_config, get_project_root
    cfg = load_config(config_path)
    root = get_project_root()
    tables_dir = root / cfg["outputs"]["tables_dir"]

    dfs = []
    for csv_file in tables_dir.glob("*_metrics.csv"):
        try:
            dfs.append(pd.read_csv(csv_file))
        except Exception:
            pass

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        out = tables_dir / "all_experiments_summary.csv"
        combined.to_csv(out, index=False)
        print(f"\nSummary table saved to {out}")
        cols = [c for c in ["model", "f1_macro", "auc_macro", "accuracy",
                            "precision_macro", "recall_macro"]
                if c in combined.columns]
        print("\n" + combined[cols].sort_values("f1_macro", ascending=False).to_string())


def main():
    args = parse_args()
    config = args.config

    for tag, model, clahe, loss, sampler in EXPERIMENTS:
        try:
            run_experiment(tag, model, clahe, loss, sampler, config, args.quick)
        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] Experiment {tag} failed: {e}")
            continue

    from src.utils.paths import load_config, get_project_root
    cfg = load_config(config)
    root = get_project_root()
    models_dir = root / cfg["outputs"]["models_dir"]
    try:
        run_ensemble(config, models_dir)
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] Ensemble (E6) failed: {e}")

    print("\n\n=== All experiments finished ===")
    aggregate_results(config)


if __name__ == "__main__":
    main()
