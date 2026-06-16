from __future__ import annotations

from pathlib import Path
from typing import Any

from src.build_crops.build_crops_for_cnn import build_crops


def crops_exist(crops_root: Path) -> bool:
    train_dir = crops_root / "train"
    val_dir = crops_root / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        return False
    try:
        next(train_dir.iterdir())
        next(val_dir.iterdir())
    except StopIteration:
        return False
    return True


def ensure_crops(
    *,
    yolo_yaml: Path,
    crops_root: Path,
    classes_path: Path,
    padding: float = 0.05,
    overwrite: bool = False,
) -> dict[str, Any] | None:
    if crops_exist(crops_root) and not overwrite:
        return None

    return build_crops(
        data_yaml=yolo_yaml,
        output_root=crops_root,
        classes_path=classes_path,
        padding=padding,
        overwrite=overwrite,
    )
