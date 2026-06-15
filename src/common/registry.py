from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_registry(pipeline: str, mode: str = "production") -> dict[str, Any]:
    return {
        "pipeline": pipeline,
        "mode": mode,
        "updated_at": None,
        "total_runs": 0,
        "runs": {},
        "best_per_experiment": {},
        "best_overall": None,
    }


def load_registry(path: Path, pipeline: str, mode: str = "production") -> dict[str, Any]:
    if not path.exists():
        return default_registry(pipeline, mode)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"invalid registry: {path}")
    data.setdefault("runs", {})
    data.setdefault("best_per_experiment", {})
    return data


def save_registry(path: Path, registry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    registry["updated_at"] = _utc_now()
    registry["total_runs"] = len(registry.get("runs", {}))
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")


def is_complete(registry: dict[str, Any], run_id: str) -> bool:
    entry = registry.get("runs", {}).get(run_id)
    return bool(entry and entry.get("status") == "completed")


def get_run_entry(registry: dict[str, Any], run_id: str) -> dict[str, Any] | None:
    return registry.get("runs", {}).get(run_id)


def update_run(
    registry: dict[str, Any],
    *,
    run_id: str,
    experiment: str,
    run_path: Path,
    weights_path: Path,
    status: str,
    metrics: dict[str, Any] | None = None,
    mode: str = "production",
    reused: bool = False,
) -> None:
    registry["runs"][run_id] = {
        "experiment": experiment,
        "dir": run_path.as_posix(),
        "weights": weights_path.as_posix(),
        "status": status,
        "mode": mode,
        "reused": reused,
        "metrics": metrics or {},
        "updated_at": _utc_now(),
    }


def set_experiment_best(registry: dict[str, Any], experiment: str, run_id: str) -> None:
    registry.setdefault("best_per_experiment", {})[experiment] = run_id


def set_best_overall(registry: dict[str, Any], run_id: str) -> None:
    registry["best_overall"] = run_id
