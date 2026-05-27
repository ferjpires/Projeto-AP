import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


def apply_clahe(image: Image.Image, clip_limit: float = 2.0,
                tile_grid_size: tuple = (8, 8)) -> Image.Image:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to a PIL image.
    Works on the L channel in LAB colour space for natural images.
    For grayscale fluoroscopic images, applies directly to the gray channel.
    """
    img_np = np.array(image.convert("RGB"))
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    return Image.fromarray(img_eq)


class CLAHETransform:
    """Callable wrapper for CLAHE so it can be composed with torchvision transforms."""
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        return apply_clahe(img, self.clip_limit, self.tile_grid_size)

    def __repr__(self):
        return f"CLAHETransform(clip_limit={self.clip_limit})"


# ImageNet statistics (used for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(image_size: int = 224, split: str = "train",
                   use_clahe: bool = False,
                   augmentation_cfg: dict = None) -> T.Compose:
    """
    Build torchvision transform pipelines for train / val / test.

    Args:
        image_size: Target spatial resolution (square).
        split: 'train', 'val', or 'test'.
        use_clahe: Whether to prepend CLAHE enhancement.
        augmentation_cfg: dict from config.yaml['augmentation'].
    """
    if augmentation_cfg is None:
        augmentation_cfg = {
            "random_rotation": 10,
            "horizontal_flip": True,
            "vertical_flip": False,
            "color_jitter": True,
            "random_erasing": False,
        }

    base_pre = []
    if use_clahe:
        base_pre.append(CLAHETransform())

    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if split == "train":
        strong = augmentation_cfg.get("strong_aug", False)
        crop_scale = (0.7, 1.0) if strong else (0.8, 1.0)
        aug = [
            T.RandomResizedCrop(image_size, scale=crop_scale,
                                interpolation=InterpolationMode.BILINEAR),
        ]
        if augmentation_cfg.get("horizontal_flip", True):
            aug.append(T.RandomHorizontalFlip())
        if augmentation_cfg.get("vertical_flip", False):
            aug.append(T.RandomVerticalFlip())
        rot = augmentation_cfg.get("random_rotation", 10)
        if rot > 0:
            aug.append(T.RandomRotation(degrees=rot))
        if strong:
            n_ops = augmentation_cfg.get("randaugment_n", 2)
            magnitude = augmentation_cfg.get("randaugment_m", 9)
            aug.append(T.RandAugment(num_ops=n_ops, magnitude=magnitude))
            if augmentation_cfg.get("color_jitter", True):
                aug.append(T.ColorJitter(brightness=0.2, contrast=0.2))
        elif augmentation_cfg.get("color_jitter", True):
            aug.append(T.ColorJitter(brightness=0.2, contrast=0.2,
                                      saturation=0.1, hue=0.05))
        aug += [
            T.ToTensor(),
            normalize,
        ]
        erase_p = 0.25 if strong else (0.2 if augmentation_cfg.get("random_erasing", False) else 0.0)
        if erase_p > 0:
            erase_scale = (0.02, 0.15) if strong else (0.02, 0.1)
            aug.append(T.RandomErasing(p=erase_p, scale=erase_scale))
        return T.Compose(base_pre + aug)

    else:
        # val / test: deterministic
        return T.Compose(base_pre + [
            T.Resize((image_size, image_size),
                     interpolation=InterpolationMode.BILINEAR),
            T.ToTensor(),
            normalize,
        ])
