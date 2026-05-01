"""
Model factory for ERCP classification.
Supports: ResNet18/50, EfficientNet-B0/B3, DenseNet121, MobileNetV3, DeiT-tiny.
All models use ImageNet pretrained weights by default.
"""
import torch
import torch.nn as nn
import timm
from torchvision import models
from torchvision.models import (
    ResNet18_Weights, ResNet50_Weights,
    DenseNet121_Weights, MobileNet_V3_Small_Weights,
)


NUM_CLASSES = 4
MODEL_REGISTRY = [
    "resnet18", "resnet50",
    "efficientnet_b0", "efficientnet_b3",
    "densenet121",
    "mobilenet_v3",
    "deit_tiny",
]


def build_model(model_name: str, num_classes: int = NUM_CLASSES,
                pretrained: bool = True) -> nn.Module:
    """
    Build a classification model.

    Args:
        model_name: One of the MODEL_REGISTRY names.
        num_classes: Number of output classes (4 for MIQR-CC).
        pretrained: Load ImageNet pretrained weights.

    Returns:
        nn.Module ready for fine-tuning.
    """
    model_name = model_name.lower()

    # ------------------------------------------------------------------ ResNet
    if model_name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    # --------------------------------------------------------------- DenseNet
    if model_name == "densenet121":
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    # ------------------------------------------------------------ MobileNetV3
    if model_name == "mobilenet_v3":
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    # ---------------------------------------------------------- EfficientNet (timm)
    if model_name in ("efficientnet_b0", "efficientnet_b3"):
        variant = "efficientnet_b0" if model_name == "efficientnet_b0" else "efficientnet_b3"
        model = timm.create_model(variant, pretrained=pretrained, num_classes=num_classes)
        return model

    # --------------------------------------------------------- DeiT-tiny (timm)
    if model_name == "deit_tiny":
        model = timm.create_model(
            "deit_tiny_patch16_224", pretrained=pretrained, num_classes=num_classes
        )
        return model

    raise ValueError(
        f"Unknown model '{model_name}'. Choose from: {MODEL_REGISTRY}"
    )


def freeze_backbone(model: nn.Module, model_name: str) -> nn.Module:
    """
    Freeze all backbone layers, keeping only the classifier head trainable.
    Useful for a quick feature-extraction warmup phase.
    """
    model_name = model_name.lower()

    if "resnet" in model_name:
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    elif "densenet" in model_name:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    elif "mobilenet" in model_name:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    elif "efficientnet" in model_name or "deit" in model_name:
        # timm models expose head/classifier
        for name, param in model.named_parameters():
            if "head" not in name and "classifier" not in name:
                param.requires_grad = False

    return model


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
