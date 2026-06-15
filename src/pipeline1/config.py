from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    import yaml

    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"invalid config: {path}")
    return cfg


def apply_test_mode(cfg: dict[str, Any]) -> dict[str, Any]:
    patched = deepcopy(cfg)
    test_cfg = patched.get("test", {})
    patched["baseline"]["epochs"] = int(test_cfg.get("epochs", 1))
    patched["baseline"]["patience"] = int(test_cfg.get("patience", 1))
    patched["baseline"]["workers"] = int(test_cfg.get("workers", 0))
    if "batch" in test_cfg:
        patched["baseline"]["batch"] = int(test_cfg["batch"])
    patched["output"] = patched.get("test_output", f"{patched['output']}_test")
    patched["mode"] = "test"
    return patched


def output_root(cfg: dict[str, Any]) -> Path:
    return Path(cfg["output"])


def data_yaml_path(cfg: dict[str, Any]) -> Path:
    return Path(cfg["data"])


def experiment_order(cfg: dict[str, Any]) -> list[str]:
    return list(cfg.get("experiment_order", []))
