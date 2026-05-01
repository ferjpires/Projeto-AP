import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    Reduces loss contribution of easy examples, focusing on hard ones.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """
    def __init__(self, gamma: float = 2.0,
                 weight: Optional[torch.Tensor] = None,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight,
                                  reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def build_loss(config: dict, class_weights: Optional[torch.Tensor] = None,
               device: torch.device = None) -> nn.Module:
    """
    Instantiate the loss function from config.

    Args:
        config: Full config dict.
        class_weights: Tensor of per-class weights (from training set).
        device: torch.device to move weights to.
    """
    loss_cfg = config.get("loss", {})
    loss_name = loss_cfg.get("name", "weighted_cross_entropy")

    if class_weights is not None and device is not None:
        class_weights = class_weights.to(device)

    if loss_name == "weighted_cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights)

    elif loss_name == "focal":
        gamma = loss_cfg.get("focal_gamma", 2.0)
        return FocalLoss(gamma=gamma, weight=class_weights)

    else:
        raise ValueError(f"Unknown loss: {loss_name}. Choose 'weighted_cross_entropy' or 'focal'.")
