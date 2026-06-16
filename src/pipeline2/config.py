from __future__ import annotations

import json
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


def crops_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg["data"]["crops_dir"])


def classes_path(cfg: dict[str, Any]) -> Path:
    return Path(cfg["data"]["classes"])


def yolo_yaml_path(cfg: dict[str, Any]) -> Path:
    return Path(cfg["data"]["yolo_yaml"])


def yolo_best_source(cfg: dict[str, Any]) -> Path:
    return Path(cfg["yolo"]["source"])


def experiment_order(cfg: dict[str, Any]) -> list[str]:
    return list(cfg.get("experiment_order", []))


def load_pipeline1_best(source: Path) -> dict[str, Any]:
    if not source.exists():
        raise FileNotFoundError(
            f"Pipeline 1 best not found: {source} — run `python -m src.pipeline1` first"
        )

    payload = json.loads(source.read_text(encoding="utf-8"))
    weights = Path(payload["weights"])
    if not weights.exists():
        raise FileNotFoundError(
            f"YOLO weights from Pipeline 1 not found: {weights} — "
            "re-run or restore Pipeline 1 weights"
        )
    return payload
