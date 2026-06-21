from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BestModelPaths:
    yolo_weights: Path
    yolo_run_id: str
    yolo_imgsz: int
    cnn_weights: Path | None = None
    cnn_run_id: str | None = None


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_best_models(project_root: Path) -> BestModelPaths:
    p1_best_path = project_root / "artifacts" / "pipeline1" / "best" / "best.json"
    if not p1_best_path.exists():
        raise FileNotFoundError(f"Pipeline 1 best model not found: {p1_best_path}")

    p1_best = _read_json(p1_best_path)
    yolo_weights = project_root / p1_best["weights"]
    if not yolo_weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {yolo_weights}")

    yolo_imgsz = int(p1_best.get("hyperparams", {}).get("imgsz", 640))

    p2_best_path = project_root / "artifacts" / "pipeline2" / "best" / "best.json"
    cnn_weights: Path | None = None
    cnn_run_id: str | None = None

    if p2_best_path.exists():
        p2_best = _read_json(p2_best_path)
        cnn_weights = project_root / p2_best["weights"]
        cnn_run_id = p2_best.get("run_id")
        if not cnn_weights.exists():
            raise FileNotFoundError(f"CNN weights not found: {cnn_weights}")

    return BestModelPaths(
        yolo_weights=yolo_weights,
        yolo_run_id=p1_best["run_id"],
        yolo_imgsz=yolo_imgsz,
        cnn_weights=cnn_weights,
        cnn_run_id=cnn_run_id,
    )
