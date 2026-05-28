"""
Grad-CAM implementation for ERCP classification models.
Works with ResNet, DenseNet, EfficientNet, MobileNet, DeiT.

Uses pytorch-grad-cam library for robustness.
Falls back to a manual implementation if the library is unavailable.
"""
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2


CLASS_NAMES = ["Biliary_Leaks", "Lithiasis", "Normal", "Stricture"]


# ---------------------------------------------------------------------------
# Target layer selection (model-specific)
# ---------------------------------------------------------------------------

def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """Return the last convolutional layer suitable for Grad-CAM."""
    model_name = model_name.lower()

    if "resnet" in model_name:
        return model.layer4[-1]

    elif "densenet" in model_name:
        return model.features.denseblock4.denselayer16.conv2

    elif "efficientnet" in model_name:
        # timm EfficientNet
        return model.conv_head

    elif "mobilenet" in model_name:
        return model.features[-1][0]  # last conv in features

    elif "deit" in model_name:
        # For ViT/DeiT use the last attention block norm
        return model.blocks[-1].norm1

    elif "convnext" in model_name:
        # timm ConvNeXt: last block of the last stage
        return model.stages[-1].blocks[-1]

    raise ValueError(f"Cannot determine target layer for model: {model_name}")


# ---------------------------------------------------------------------------
# Grad-CAM using pytorch-grad-cam library
# ---------------------------------------------------------------------------

def _gradcam_with_library(model, target_layer, img_tensor, target_class,
                           device) -> np.ndarray:
    """Use pytorch-grad-cam library."""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
        grayscale_cam = cam(input_tensor=img_tensor.unsqueeze(0).to(device),
                            targets=targets)
    return grayscale_cam[0]


# ---------------------------------------------------------------------------
# Manual Grad-CAM fallback
# ---------------------------------------------------------------------------

class _GradCAMManual:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self._hooks = []
        self._hooks.append(
            target_layer.register_forward_hook(self._save_activation)
        )
        self._hooks.append(
            target_layer.register_full_backward_hook(self._save_gradient)
        )

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, img_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(img_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        score = output[0, target_class]
        score.backward()

        weights = self.gradients.mean(dim=(-1, -2), keepdim=True)  # (B, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1).squeeze()     # (H, W)
        cam = torch.relu(cam).cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_gradcam(model: nn.Module, img_tensor: torch.Tensor,
                    target_layer: nn.Module, target_class: Optional[int],
                    device: torch.device) -> np.ndarray:
    """
    Compute a Grad-CAM heatmap.

    Returns:
        heatmap: (H, W) float array in [0, 1].
    """
    try:
        return _gradcam_with_library(model, target_layer, img_tensor,
                                     target_class, device)
    except Exception:
        gcam = _GradCAMManual(model, target_layer)
        img_tensor = img_tensor.unsqueeze(0).to(device)
        heatmap = gcam(img_tensor, target_class)
        gcam.remove_hooks()
        return heatmap


def overlay_heatmap(original_img: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.4, colormap: int = cv2.COLORMAP_JET
                    ) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on the original image.

    Args:
        original_img: (H, W, 3) uint8 RGB image.
        heatmap: (H, W) float in [0, 1].
        alpha: Transparency of the heatmap overlay.
        colormap: OpenCV colormap.

    Returns:
        (H, W, 3) uint8 RGB overlay.
    """
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap_rgb, alpha, 0)
    return overlay


def save_gradcam_figure(original: np.ndarray, heatmap: np.ndarray,
                        save_path: Path,
                        true_label: str, pred_label: str,
                        probabilities: np.ndarray,
                        class_names: List[str] = None) -> None:
    """Save a 3-panel figure: original | heatmap | overlay."""
    class_names = class_names or CLASS_NAMES
    overlay = overlay_heatmap(original, heatmap)
    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * cv2.resize(heatmap, (original.shape[1], original.shape[0]))),
        cv2.COLORMAP_JET
    )
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].imshow(original)
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(heatmap_rgb)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    match = "✓" if true_label == pred_label else "✗"
    axes[2].set_title(f"{match} True: {true_label}\nPred: {pred_label}", fontsize=11)
    axes[2].axis("off")

    # Probability bar
    bar_ax = fig.add_axes([0.35, 0.02, 0.30, 0.12])
    colors = ["#e74c3c" if c == pred_label else "#95a5a6" for c in class_names]
    bar_ax.barh(class_names, probabilities, color=colors)
    bar_ax.set_xlim(0, 1)
    bar_ax.set_xlabel("Probability", fontsize=8)
    bar_ax.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0.18, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_gradcam_examples(model: nn.Module, dataset,
                              model_name: str, config: dict,
                              n_per_class: int = 3) -> None:
    """
    Generate Grad-CAM visualizations for n_per_class images from each class.
    Saves correct and (if available) incorrect predictions.
    """
    from src.data.transforms import get_transforms

    device_str = config["training"].get("device", "cpu")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    class_names = config["data"]["class_names"]
    gradcam_dir = Path(config["outputs"]["gradcam_dir"])

    try:
        target_layer = get_target_layer(model, model_name)
    except Exception as e:
        print(f"  [WARN] Could not determine target layer: {e}")
        return

    # Inverse-normalize transform to recover original image
    inv_mean = [-m / s for m, s in zip([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])]
    inv_std = [1 / s for s in [0.229, 0.224, 0.225]]
    inv_norm = __import__("torchvision").transforms.Normalize(
        mean=inv_mean, std=inv_std
    )

    # Index samples by class
    by_class = {i: [] for i in range(len(class_names))}
    for idx, (_, label) in enumerate(dataset.samples):
        by_class[label].append(idx)

    for cls_idx, cls_name in enumerate(class_names):
        indices = by_class[cls_idx][:n_per_class * 3]  # extra for failed preds
        saved = 0
        for sample_idx in indices:
            if saved >= n_per_class:
                break
            img_tensor, true_label = dataset[sample_idx]

            with torch.no_grad():
                out = model(img_tensor.unsqueeze(0).to(device))
                probs = torch.softmax(out, dim=1)[0].cpu().numpy()
                pred_label = int(probs.argmax())

            heatmap = compute_gradcam(
                model, img_tensor, target_layer, pred_label, device
            )

            # Recover original image
            orig_tensor = inv_norm(img_tensor).permute(1, 2, 0).numpy()
            orig_tensor = np.clip(orig_tensor, 0, 1)
            original = (orig_tensor * 255).astype(np.uint8)

            suffix = "correct" if pred_label == true_label else "wrong"
            save_path = (gradcam_dir / cls_name /
                         f"sample_{saved:02d}_{suffix}.png")

            save_gradcam_figure(
                original, heatmap, save_path,
                true_label=class_names[true_label],
                pred_label=class_names[pred_label],
                probabilities=probs,
                class_names=class_names,
            )
            saved += 1

        print(f"  Grad-CAM: {cls_name} — {saved} examples saved")
