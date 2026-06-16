from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.common.memory import purge_ultralytics_run_dir, release_runtime_memory
from src.common.yolo_data import prepare_data_yaml


def resolve_model_checkpoint(cfg: dict[str, Any], model_name: str) -> str:
    weights_map = cfg.get("model_weights", {})
    if model_name in weights_map:
        return str(weights_map[model_name])
    return f"{model_name}.pt"


def train_yolo_run(
    *,
    params: dict[str, Any],
    run_dir: Path,
    weights_path: Path,
    data_yaml: Path,
    cfg: dict[str, Any],
    device: str | None,
) -> None:
    from ultralytics import YOLO

    run_dir.mkdir(parents=True, exist_ok=True)
    prepared_data = prepare_data_yaml(data_yaml.resolve())
    ultra_dir: Path | None = None
    model = None

    try:
        model = YOLO(resolve_model_checkpoint(cfg, params["model"]))
        model.train(
            data=str(prepared_data),
            epochs=int(params["epochs"]),
            imgsz=int(params["imgsz"]),
            batch=int(params["batch"]),
            lr0=float(params["lr"]),
            mosaic=float(params["mosaic"]),
            device=device,
            project=str(run_dir.resolve()),
            name="_ultra",
            patience=int(params["patience"]),
            seed=int(params["seed"]),
            workers=int(params["workers"]),
            exist_ok=True,
            verbose=False,
            plots=False,
        )

        trainer = model.trainer
        if trainer is None or trainer.save_dir is None:
            raise RuntimeError("training finished without a save directory")
        ultra_dir = Path(trainer.save_dir)

        best_src = ultra_dir / "weights" / "best.pt"
        if not best_src.exists():
            best_src = ultra_dir / "weights" / "last.pt"
        if not best_src.exists():
            raise FileNotFoundError(f"no weights found under {ultra_dir}")

        shutil.copy2(best_src, weights_path)

        results_csv = ultra_dir / "results.csv"
        if results_csv.exists():
            shutil.copy2(results_csv, run_dir / "train_log.csv")
    finally:
        if ultra_dir is not None:
            purge_ultralytics_run_dir(ultra_dir)
        if model is not None and hasattr(model, "trainer"):
            model.trainer = None
        release_runtime_memory(model)


def write_run_config(
    run_dir: Path,
    *,
    run_id: str,
    experiment: str,
    params: dict[str, Any],
    device: str | None,
    status: str,
    mode: str,
    started_at: str,
    finished_at: str | None = None,
    reused: bool = False,
) -> None:
    payload = {
        "run_id": run_id,
        "experiment": experiment,
        "pipeline": "pipeline1",
        "mode": mode,
        "reused": reused,
        "hyperparams": params,
        "device": device,
        "started_at": started_at,
        "finished_at": finished_at,
        "status": status,
    }
    path = run_dir / "run_config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def write_metrics(
    run_dir: Path,
    *,
    run_id: str,
    experiment: str,
    weights_file: str,
    metrics: dict[str, Any],
    reused: bool = False,
) -> Path:
    payload = {
        "run_id": run_id,
        "experiment": experiment,
        "weights_file": weights_file,
        "reused": reused,
        "metrics": metrics,
    }
    path = run_dir / "metrics.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
