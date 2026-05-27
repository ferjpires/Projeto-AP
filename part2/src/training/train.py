"""
Training loop for ERCP classification models.
"""
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np


def train_one_epoch(model: nn.Module, loader: DataLoader,
                    criterion: nn.Module, optimizer: torch.optim.Optimizer,
                    device: torch.device, scaler=None) -> dict:
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, labels in tqdm(loader, desc="  Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    n = len(loader.dataset)
    avg_loss = total_loss / n
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return {"loss": avg_loss, "f1_macro": f1}


@torch.no_grad()
def validate_one_epoch(model: nn.Module, loader: DataLoader,
                       criterion: nn.Module, device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, labels in tqdm(loader, desc="  Val  ", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    n = len(loader.dataset)
    avg_loss = total_loss / n
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return {"loss": avg_loss, "f1_macro": f1}


def build_optimizer(model: nn.Module, config: dict,
                    lr_override: float = None) -> torch.optim.Optimizer:
    lr = lr_override if lr_override is not None else config["training"]["learning_rate"]
    wd = config["training"].get("weight_decay", 1e-4)
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=wd
    )


def build_scheduler(optimizer, config: dict, n_epochs: int):
    """
    Cosine schedule with optional linear warmup (config.training.warmup_epochs).
    """
    sched_name = config["training"].get("scheduler", "cosine")
    warmup = config["training"].get("warmup_epochs", 0)
    if sched_name == "cosine":
        if warmup > 0:
            def _lr(epoch):
                if epoch < warmup:
                    return float(epoch + 1) / float(warmup + 1)
                progress = (epoch - warmup) / max(1, n_epochs - warmup)
                import math
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    elif sched_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )
    return None


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                criterion: nn.Module, config: dict,
                experiment_dir: Path, model_name: str = "model") -> dict:
    """
    Full training loop with early stopping, LR scheduling, and checkpoint saving.

    Returns history dict with loss/f1 curves.
    """
    device_str = config["training"].get("device", "cpu")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("  [WARNING] CUDA requested but not available — using CPU.")
    model = model.to(device)
    criterion = criterion.to(device) if hasattr(criterion, "to") else criterion

    n_epochs = config["training"]["num_epochs"]
    patience = config["training"].get("early_stopping_patience", 7)
    head_warmup = config["training"].get("head_warmup_epochs", 0)
    head_lr = config["training"].get("head_lr", 1e-3)

    # optional head-only warmup with frozen backbone
    if head_warmup > 0:
        from src.models.build_model import freeze_backbone
        freeze_backbone(model, model_name)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  [Stage 1] Head-only warmup: {head_warmup} eps, LR={head_lr}, "
              f"trainable params={n_trainable:,}")

    initial_lr = head_lr if head_warmup > 0 else config["training"]["learning_rate"]
    optimizer = build_optimizer(model, config, lr_override=initial_lr)
    # Scheduler covers the post-warmup phase only
    scheduler = build_scheduler(optimizer, config, max(1, n_epochs - head_warmup))

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = experiment_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    best_ckpt = ckpt_dir / "best_model.pth"

    history = {
        "train_loss": [], "val_loss": [],
        "train_f1": [], "val_f1": [],
    }
    best_val_f1 = -1.0
    epochs_no_improve = 0

    print(f"\n{'='*55}")
    print(f"  Training: {model_name}  |  device: {device}")
    print(f"  Epochs: {n_epochs}  |  LR: {config['training']['learning_rate']}")
    print(f"{'='*55}")

    for epoch in range(1, n_epochs + 1):
        if head_warmup > 0 and epoch == head_warmup + 1:
            for p in model.parameters():
                p.requires_grad = True
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  [Stage 2] Unfreezing all layers. Trainable params={n_trainable:,}, "
                  f"LR={config['training']['learning_rate']}")
            optimizer = build_optimizer(model, config)
            scheduler = build_scheduler(optimizer, config, n_epochs - head_warmup)

        t0 = time.time()
        train_stats = train_one_epoch(model, train_loader, criterion,
                                      optimizer, device, scaler)
        val_stats = validate_one_epoch(model, val_loader, criterion, device)

        if scheduler is not None and epoch > head_warmup:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_stats["f1_macro"])
            else:
                scheduler.step()

        history["train_loss"].append(train_stats["loss"])
        history["val_loss"].append(val_stats["loss"])
        history["train_f1"].append(train_stats["f1_macro"])
        history["val_f1"].append(val_stats["f1_macro"])

        elapsed = time.time() - t0
        print(f"  Epoch {epoch:3d}/{n_epochs} | "
              f"Train loss={train_stats['loss']:.4f} f1={train_stats['f1_macro']:.4f} | "
              f"Val loss={val_stats['loss']:.4f} f1={val_stats['f1_macro']:.4f} | "
              f"{elapsed:.1f}s")

        # Save best
        if val_stats["f1_macro"] > best_val_f1:
            best_val_f1 = val_stats["f1_macro"]
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1_macro": best_val_f1,
                "config": config,
                "model_name": model_name,
            }, best_ckpt)
            print(f"    ✓ Best model saved (val F1={best_val_f1:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    print(f"\n  Training complete. Best val F1 macro = {best_val_f1:.4f}")
    return history, best_ckpt
