from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

from src.common.memory import release_runtime_memory
from src.common.run_naming import build_run_id, run_directory, weights_filename
from src.pipeline2.config import classes_path, crops_dir, output_root
from src.pipeline2.train import train_cnn_run, utc_now, write_metrics, write_run_config


def execute_single_run(
    *,
    cfg: dict[str, Any],
    spec: dict[str, Any],
    device: str | None,
    skip_eval: bool,
) -> dict[str, Any]:
    experiment = spec["experiment"]
    params = spec["params"]
    run_id = spec["run_id"]
    run_dir = Path(spec["run_dir"])
    weights_path = Path(spec["weights_path"])
    crops_root = crops_dir(cfg)
    classes_file = classes_path(cfg)
    mode = cfg.get("mode", "production")

    started_at = utc_now()
    try:
        metrics, _history = train_cnn_run(
            params=params,
            run_dir=run_dir,
            weights_path=weights_path,
            crops_root=crops_root,
            classes_path=classes_file,
            device=device,
        )

        write_metrics(
            run_dir,
            run_id=run_id,
            experiment=experiment,
            weights_file=weights_path.name,
            metrics=metrics,
            reused=False,
        )
        write_run_config(
            run_dir,
            run_id=run_id,
            experiment=experiment,
            params=params,
            device=device,
            status="completed",
            mode=mode,
            started_at=started_at,
            finished_at=utc_now(),
            reused=False,
        )

        return {
            "run_id": run_id,
            "experiment": experiment,
            "dir": run_dir.as_posix(),
            "weights": weights_path.as_posix(),
            "hyperparams": params,
            "metrics": metrics,
            "reused": False,
        }
    except Exception as exc:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write_run_config(
            run_dir,
            run_id=run_id,
            experiment=experiment,
            params=params,
            device=device,
            status="failed",
            mode=mode,
            started_at=started_at,
            finished_at=utc_now(),
            reused=False,
        )
        raise exc
    finally:
        release_runtime_memory()


def build_run_spec_dict(
    cfg: dict[str, Any],
    experiment: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    run_id = build_run_id(params, pipeline="pipeline2")
    run_dir = run_directory(output_root(cfg), experiment, run_id)
    weights_path = run_dir / weights_filename(run_id, pipeline="pipeline2")
    return {
        "experiment": experiment,
        "params": params,
        "run_id": run_id,
        "run_dir": run_dir.as_posix(),
        "weights_path": weights_path.as_posix(),
    }


def write_job_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def read_job_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
