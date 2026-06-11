from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    import torch
    from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def yolo_norm_to_xyxy(
    cx: float,
    cy: float,
    width: float,
    height: float,
    img_width: int,
    img_height: int,
) -> tuple[float, float, float, float]:
    x1 = (cx - width / 2) * img_width
    y1 = (cy - height / 2) * img_height
    x2 = (cx + width / 2) * img_width
    y2 = (cy + height / 2) * img_height
    return x1, y1, x2, y2


def apply_bbox_padding(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_width: int,
    img_height: int,
    padding: float = 0.05,
) -> tuple[int, int, int, int]:
    box_w = max(x2 - x1, 1.0)
    box_h = max(y2 - y1, 1.0)
    pad_x = box_w * padding
    pad_y = box_h * padding

    left = int(max(0, np.floor(x1 - pad_x)))
    top = int(max(0, np.floor(y1 - pad_y)))
    right = int(min(img_width, np.ceil(x2 + pad_x)))
    bottom = int(min(img_height, np.ceil(y2 + pad_y)))

    if right <= left:
        right = min(img_width, left + 1)
    if bottom <= top:
        bottom = min(img_height, top + 1)
    return left, top, right, bottom


def crop_bbox(
    image: Image.Image | np.ndarray,
    bbox_xyxy: tuple[float, float, float, float] | list[float],
    padding: float = 0.05,
) -> Image.Image:
    if isinstance(image, np.ndarray):
        # Ultralytics uses BGR numpy arrays.
        image = Image.fromarray(image[:, :, ::-1])

    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    left, top, right, bottom = apply_bbox_padding(
        x1, y1, x2, y2, image.width, image.height, padding=padding
    )
    return image.crop((left, top, right, bottom))


def build_cnn_transform(input_size: int = 224, train: bool = False) -> transforms.Compose:
    from torchvision import transforms

    if train:
        return transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def preprocess_crop_for_cnn(
    crop: Image.Image,
    transform: transforms.Compose,
    device: torch.device,
) -> torch.Tensor:
    import torch

    tensor = transform(crop.convert("RGB")).unsqueeze(0)
    return tensor.to(device)
