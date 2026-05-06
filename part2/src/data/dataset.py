from pathlib import Path
from typing import Callable, Optional, Tuple, List, Dict

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np


CLASS_NAMES = ["Biliary_Leaks", "Lithiasis", "Normal", "Stricture"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


class ERCPDataset(Dataset):
    """
    Loads ERCP images organised as:
        root/
          Biliary_Leaks/*.png
          Lithiasis/*.png
          Normal/*.png
          Stricture/*.png
    """

    def __init__(self, root_dir: str, transform: Optional[Callable] = None,
                 class_names: List[str] = None):
        self.root = Path(root_dir)
        self.transform = transform
        self.class_names = class_names or CLASS_NAMES
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self):
        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        for cls in self.class_names:
            cls_dir = self.root / cls
            if not cls_dir.exists():
                print(f"  [WARNING] Class directory not found: {cls_dir}")
                continue
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in extensions:
                    self.samples.append((img_path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_class_counts(self) -> Dict[str, int]:
        counts = {c: 0 for c in self.class_names}
        for _, label in self.samples:
            counts[self.class_names[label]] += 1
        return counts

    def get_class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for CrossEntropyLoss."""
        counts = self.get_class_counts()
        total = sum(counts.values())
        n_cls = len(self.class_names)
        weights = []
        for cls in self.class_names:
            c = counts[cls]
            w = total / (n_cls * c) if c > 0 else 0.0
            weights.append(w)
        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_weights(self) -> torch.Tensor:
        """Per-sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights()
        return torch.tensor([class_weights[label] for _, label in self.samples])


def build_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader,
                                             ERCPDataset, ERCPDataset, ERCPDataset]:
    """Build train / val / test DataLoaders from config."""
    from src.data.transforms import get_transforms

    aug_cfg = config.get("augmentation", {})
    use_clahe = aug_cfg.get("use_clahe", False)
    img_size = config["training"]["image_size"]
    batch = config["training"]["batch_size"]
    n_workers = config["training"].get("num_workers", 4)
    class_names = config["data"]["class_names"]
    use_sampler = config.get("sampling", {}).get("use_weighted_sampler", False)
    pin = torch.cuda.is_available()

    train_ds = ERCPDataset(
        config["data"]["train_dir"],
        transform=get_transforms(img_size, "train", use_clahe, aug_cfg),
        class_names=class_names,
    )
    val_ds = ERCPDataset(
        config["data"]["val_dir"],
        transform=get_transforms(img_size, "val", use_clahe),
        class_names=class_names,
    )
    test_ds = ERCPDataset(
        config["data"]["test_dir"],
        transform=get_transforms(img_size, "test", use_clahe),
        class_names=class_names,
    )

    if use_sampler:
        sample_weights = train_ds.get_sample_weights()
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),
                                        replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch, sampler=sampler,
                                  num_workers=n_workers, pin_memory=pin)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                                  num_workers=n_workers, pin_memory=pin)

    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False,
                            num_workers=n_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False,
                             num_workers=n_workers, pin_memory=pin)

    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds
