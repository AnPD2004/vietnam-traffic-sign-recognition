from __future__ import annotations

import re
from pathlib import Path

DEFAULT_CLASSES_PATH = Path("data/vn-traffic-signs/classes.txt")

_CLASS_NAMES: list[str] | None = None
_NAME_TO_ID: dict[str, int] | None = None


def load_class_names(classes_path: Path | str = DEFAULT_CLASSES_PATH) -> list[str]:
    path = Path(classes_path)
    if not path.exists():
        raise FileNotFoundError(f"classes file not found: {path}")

    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ValueError(f"classes file is empty: {path}")
    return names


def get_class_names(classes_path: Path | str | None = None) -> list[str]:
    global _CLASS_NAMES, _NAME_TO_ID
    if _CLASS_NAMES is None or classes_path is not None:
        _CLASS_NAMES = load_class_names(classes_path or DEFAULT_CLASSES_PATH)
        _NAME_TO_ID = {name: idx for idx, name in enumerate(_CLASS_NAMES)}
    return _CLASS_NAMES


def get_class_name(class_id: int, classes_path: Path | str | None = None) -> str:
    names = get_class_names(classes_path)
    if class_id < 0 or class_id >= len(names):
        raise IndexError(f"class_id {class_id} out of range [0, {len(names) - 1}]")
    return names[class_id]


def get_class_id(class_name: str, classes_path: Path | str | None = None) -> int:
    get_class_names(classes_path)
    assert _NAME_TO_ID is not None
    if class_name not in _NAME_TO_ID:
        raise KeyError(f"unknown class name: {class_name}")
    return _NAME_TO_ID[class_name]


def sanitize_class_dir(class_name: str) -> str:
    """Filesystem-safe folder name while keeping class text readable."""
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", class_name)
    sanitized = sanitized.strip().rstrip(".")
    return sanitized or "unknown"


def class_dir_name(class_id: int, classes_path: Path | str | None = None) -> str:
    """Folder name with zero-padded id so ImageFolder sort order matches class_id."""
    name = sanitize_class_dir(get_class_name(class_id, classes_path))
    return f"{class_id:03d}_{name}"


def class_id_from_dir_name(dir_name: str) -> int:
    prefix = dir_name.split("_", 1)[0]
    return int(prefix)
