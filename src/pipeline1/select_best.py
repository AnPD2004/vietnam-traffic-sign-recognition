from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from src.common.metrics import select_best_run


def collect_experiment_entries(
    run_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for spec in run_specs:
        metrics_path = spec["run_dir"] / "metrics.json"
        if not metrics_path.exists():
            continue
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        entries.append(
            {
                "run_id": spec["run_id"],
                "experiment": spec["experiment"],
                "dir": spec["run_dir"].as_posix(),
                "weights": spec["weights_path"].as_posix(),
                "hyperparams": spec["params"],
                "metrics": payload.get("metrics", {}),
                "reused": payload.get("reused", False),
            }
        )
    return entries


def select_experiment_best(
    run_specs: list[dict[str, Any]],
    metric: str,
) -> dict[str, Any]:
    entries = collect_experiment_entries(run_specs)
    if not entries:
        raise RuntimeError("no completed runs with metrics for experiment")
    return select_best_run(entries, metric)


def write_experiment_best(
    output_root: Path,
    experiment: str,
    best_entry: dict[str, Any],
    metric: str,
) -> Path:
    metric_value = (best_entry.get("metrics") or {}).get(metric)
    payload = {
        "experiment": experiment,
        "run_id": best_entry["run_id"],
        "weights": best_entry["weights"],
        "metric_used": metric,
        "metric_value": metric_value,
        "hyperparams": best_entry.get("hyperparams"),
        "reused": best_entry.get("reused", False),
    }
    path = output_root / "runs" / experiment / "_best.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def finalize_best(
    output_root: Path,
    best_entry: dict[str, Any],
    mode: str,
) -> Path:
    best_dir = output_root / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    src_weights = Path(best_entry["weights"])
    dst_weights = best_dir / src_weights.name
    shutil.copy2(src_weights, dst_weights)

    payload = {
        "pipeline": "pipeline1",
        "mode": mode,
        "run_id": best_entry["run_id"],
        "weights": dst_weights.as_posix(),
        "source_run_dir": best_entry["dir"],
        "hyperparams": best_entry.get("hyperparams", {}),
        "metrics": best_entry.get("metrics", {}),
    }
    best_json = best_dir / "best.json"
    best_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return best_json
