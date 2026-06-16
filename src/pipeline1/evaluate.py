from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.common.memory import release_runtime_memory
from src.common.yolo_data import prepare_data_yaml


def evaluate_yolo_run(
    *,
    weights_path: Path,
    data_yaml: Path,
    params: dict[str, Any],
    device: str | None,
    skip_fps: bool = False,
) -> dict[str, Any]:
    from ultralytics import YOLO

    prepared_data = prepare_data_yaml(data_yaml.resolve())
    model = None

    try:
        model = YOLO(str(weights_path))
        results = model.val(
            data=str(prepared_data),
            imgsz=int(params["imgsz"]),
            device=device,
            verbose=False,
        )

        metrics: dict[str, Any] = {
            "precision": _safe_float(getattr(results.box, "mp", None)),
            "recall": _safe_float(getattr(results.box, "mr", None)),
            "map50": _safe_float(getattr(results.box, "map50", None)),
            "map50_95": _safe_float(getattr(results.box, "map", None)),
            "model_size_mb": round(weights_path.stat().st_size / (1024 * 1024), 4),
        }

        if not skip_fps:
            metrics.update(measure_fps(model, imgsz=int(params["imgsz"])))

        return metrics
    finally:
        if model is not None and hasattr(model, "trainer"):
            model.trainer = None
        release_runtime_memory(model)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def measure_fps(model: Any, imgsz: int, warmup: int = 3, runs: int = 10) -> dict[str, float]:
    import numpy as np

    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)

    for _ in range(warmup):
        model.predict(dummy, imgsz=imgsz, verbose=False)

    start = time.perf_counter()
    for _ in range(runs):
        model.predict(dummy, imgsz=imgsz, verbose=False)
    elapsed = time.perf_counter() - start

    fps = runs / elapsed if elapsed > 0 else 0.0
    inference_ms = (elapsed / runs) * 1000.0
    return {
        "fps": round(fps, 4),
        "inference_ms": round(inference_ms, 4),
    }
