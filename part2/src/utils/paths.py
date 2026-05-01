from pathlib import Path
import yaml


def get_project_root() -> Path:
    """Return the absolute path to the project root."""
    return Path(__file__).resolve().parent.parent.parent


def load_config(config_path: str = None) -> dict:
    """Load YAML config file."""
    if config_path is None:
        config_path = get_project_root() / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def ensure_dirs(config: dict) -> None:
    """Create all output directories specified in config."""
    dirs = [
        config["outputs"]["models_dir"],
        config["outputs"]["figures_dir"],
        config["outputs"]["confusion_matrices_dir"],
        config["outputs"]["roc_dir"],
        config["outputs"]["gradcam_dir"],
        config["outputs"]["tables_dir"],
        config["outputs"]["experiments_dir"],
    ]
    root = get_project_root()
    for d in dirs:
        (root / d).mkdir(parents=True, exist_ok=True)
    # gradcam subdirs per class
    for cls in config["data"]["class_names"]:
        (root / config["outputs"]["gradcam_dir"] / cls).mkdir(parents=True, exist_ok=True)
