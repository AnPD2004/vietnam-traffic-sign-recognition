from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

from src.common.image_ops import build_cnn_transform, preprocess_crop_for_cnn


def resolve_torch_device(device: str | None) -> torch.device:
    """Map Ultralytics-style device strings (e.g. \"0\") to torch.device."""
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.isdigit():
        if torch.cuda.is_available():
            return torch.device(f"cuda:{device}")
        return torch.device("cpu")
    return torch.device(device)


def build_classifier(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    weights = "DEFAULT" if pretrained else None

    if model_name == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    raise ValueError(f"unsupported CNN model: {model_name}")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    class_names: list[str],
    model_name: str,
    input_size: int,
    metrics: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "class_to_idx": {name: idx for idx, name in enumerate(class_names)},
        "model_name": model_name,
        "input_size": input_size,
        "metrics": metrics or {},
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    device: torch.device | str | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"CNN checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location=device or "cpu", weights_only=False)
    class_names = payload["class_names"]
    model_name = payload["model_name"]
    input_size = int(payload["input_size"])

    model = build_classifier(model_name, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    if device is not None:
        model.to(device)

    metadata = {
        "class_names": class_names,
        "class_to_idx": payload.get("class_to_idx", {name: i for i, name in enumerate(class_names)}),
        "model_name": model_name,
        "input_size": input_size,
        "metrics": payload.get("metrics", {}),
    }
    return model, metadata


def save_class_index(path: Path, class_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "class_names": class_names,
        "class_to_idx": {name: idx for idx, name in enumerate(class_names)},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class CnnClassifier:
    def __init__(self, checkpoint_path: str | Path, device: str | None = None) -> None:
        self.device = resolve_torch_device(device)
        self.model, self.metadata = load_checkpoint(checkpoint_path, device=self.device)
        self.class_names: list[str] = self.metadata["class_names"]
        self.input_size: int = int(self.metadata["input_size"])
        self.transform = build_cnn_transform(self.input_size, train=False)

    def predict_crop(self, crop: Image.Image) -> tuple[int, str, float]:
        tensor = preprocess_crop_for_cnn(crop, self.transform, self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            confidence, class_id = torch.max(probs, dim=0)

        class_index = int(class_id.item())
        if class_index < 0 or class_index >= len(self.class_names):
            raise IndexError(f"CNN predicted class_id {class_index} out of range")
        class_name = self.class_names[class_index]
        return class_index, class_name, float(confidence.item())
