from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.memory import release_runtime_memory
from src.common.registry import (
    get_run_entry,
    is_complete,
    load_registry,
    save_registry,
    set_best_overall,
    set_experiment_best,
    update_run,
)
from src.common.run_naming import build_run_id, run_directory, weights_filename
from src.pipeline2.config import (
    apply_test_mode,
    classes_path,
    crops_dir,
    experiment_order,
    load_config,
    load_pipeline1_best,
    output_root,
    yolo_best_source,
    yolo_yaml_path,
)
from src.pipeline2.experiments import build_runs_for_experiment, state_from_params
from src.pipeline2.prepare_data import crops_exist, ensure_crops
from src.pipeline2.report import generate_report
from src.pipeline2.run_one import build_run_spec_dict, write_job_file
from src.pipeline2.select_best import finalize_best, select_experiment_best, write_experiment_best
from src.pipeline2.train import utc_now, write_metrics, write_run_config


@dataclass
class RunSpec:
    experiment: str
    params: dict[str, Any]
    run_id: str
    run_dir: Path
    weights_path: Path


@dataclass
class PipelineFlags:
    config_path: Path
    test: bool
    dry_run: bool
    force: bool
    device: str | None
    skip_eval: bool
    skip_crops: bool
    experiment: str | None
    run_id: str | None
    list_runs: bool
    in_process: bool


def _load_prior_state(out_root: Path, experiment: str, order: list[str]) -> dict[str, Any]:
    idx = order.index(experiment)
    if idx == 0:
        return {}
    prev_exp = order[idx - 1]
    best_path = out_root / "runs" / prev_exp / "_best.json"
    if not best_path.exists():
        raise FileNotFoundError(
            f"cannot run {experiment} without prior results — missing {best_path}"
        )
    payload = json.loads(best_path.read_text(encoding="utf-8"))
    return state_from_params(payload["hyperparams"])


def _plan_all_specs(
    cfg: dict[str, Any],
    *,
    experiment_filter: str | None,
    run_id_filter: str | None,
) -> list[RunSpec]:
    specs: list[RunSpec] = []
    state: dict[str, Any] = {}
    order = experiment_order(cfg)

    for experiment in order:
        if experiment_filter and experiment != experiment_filter:
            continue
        if experiment != order[0] and not state:
            state = _load_prior_state(output_root(cfg), experiment, order)
        param_list = build_runs_for_experiment(cfg, experiment, state)
        for params in param_list:
            run_id = build_run_id(params, pipeline="pipeline2")
            if run_id_filter and run_id != run_id_filter:
                continue
            run_dir = run_directory(output_root(cfg), experiment, run_id)
            specs.append(
                RunSpec(
                    experiment=experiment,
                    params=params,
                    run_id=run_id,
                    run_dir=run_dir,
                    weights_path=run_dir / weights_filename(run_id, pipeline="pipeline2"),
                )
            )
        if param_list:
            state = state_from_params(param_list[0])
    return specs


def _print_run_plan(specs: list[RunSpec], mode: str) -> None:
    print(f"=== PIPELINE 2 PLAN ({mode}) — {len(specs)} runs ===")
    current_exp: str | None = None
    for spec in specs:
        if spec.experiment != current_exp:
            current_exp = spec.experiment
            print(f"\n[{current_exp}]")
        print(f"  {spec.run_id}")
        print(f"    dir: {spec.run_dir}")


def _run_in_subprocess(
    spec: RunSpec,
    *,
    cfg: dict[str, Any],
    flags: PipelineFlags,
) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        job_path = Path(tmp.name)

    try:
        write_job_file(
            job_path,
            {
                "config_path": flags.config_path.as_posix(),
                "test": flags.test,
                "device": flags.device,
                "skip_eval": flags.skip_eval,
                "spec": build_run_spec_dict(cfg, spec.experiment, spec.params),
            },
        )

        cmd = [sys.executable, "-m", "src.pipeline2.run_worker", "--job", str(job_path)]
        proc = subprocess.run(cmd, check=False)

        if proc.returncode != 0:
            error_file = spec.run_dir / "error.txt"
            if error_file.exists():
                err = error_file.read_text(encoding="utf-8", errors="replace").strip()
            else:
                err = f"worker exited with code {proc.returncode}"
            raise RuntimeError(err)

        metrics_path = spec.run_dir / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"worker finished without metrics: {metrics_path}")

        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        return {
            "run_id": spec.run_id,
            "experiment": spec.experiment,
            "dir": spec.run_dir.as_posix(),
            "weights": spec.weights_path.as_posix(),
            "hyperparams": spec.params,
            "metrics": payload.get("metrics", {}),
            "reused": False,
        }
    finally:
        job_path.unlink(missing_ok=True)
        release_runtime_memory()


def _run_in_process(
    spec: RunSpec,
    *,
    cfg: dict[str, Any],
    flags: PipelineFlags,
) -> dict[str, Any]:
    from src.pipeline2.run_one import execute_single_run

    spec_dict = build_run_spec_dict(cfg, spec.experiment, spec.params)
    return execute_single_run(
        cfg=cfg,
        spec=spec_dict,
        device=flags.device,
        skip_eval=flags.skip_eval,
    )


def _execute_run(
    spec: RunSpec,
    *,
    cfg: dict[str, Any],
    registry: dict[str, Any],
    flags: PipelineFlags,
) -> dict[str, Any]:
    mode = cfg.get("mode", "production")

    if is_complete(registry, spec.run_id) and not flags.force:
        existing = get_run_entry(registry, spec.run_id)
        if existing and existing.get("metrics"):
            print(f"  [skip] {spec.run_id} (already completed)")
            spec.run_dir.mkdir(parents=True, exist_ok=True)
            write_metrics(
                spec.run_dir,
                run_id=spec.run_id,
                experiment=spec.experiment,
                weights_file=spec.weights_path.name,
                metrics=existing["metrics"],
                reused=True,
            )
            write_run_config(
                spec.run_dir,
                run_id=spec.run_id,
                experiment=spec.experiment,
                params=spec.params,
                device=flags.device,
                status="completed",
                mode=mode,
                started_at=existing.get("updated_at", utc_now()),
                finished_at=existing.get("updated_at"),
                reused=True,
            )
            return {
                "run_id": spec.run_id,
                "experiment": spec.experiment,
                "dir": spec.run_dir.as_posix(),
                "weights": existing["weights"],
                "hyperparams": spec.params,
                "metrics": existing["metrics"],
                "reused": True,
            }

    started_at = utc_now()
    print(f"  [train] {spec.run_id}")
    try:
        if flags.in_process:
            entry = _run_in_process(spec, cfg=cfg, flags=flags)
        else:
            entry = _run_in_subprocess(spec, cfg=cfg, flags=flags)

        update_run(
            registry,
            run_id=spec.run_id,
            experiment=spec.experiment,
            run_path=spec.run_dir,
            weights_path=spec.weights_path,
            status="completed",
            metrics=entry["metrics"],
            mode=mode,
            reused=False,
        )
        release_runtime_memory()
        return entry
    except Exception as exc:
        spec.run_dir.mkdir(parents=True, exist_ok=True)
        (spec.run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write_run_config(
            spec.run_dir,
            run_id=spec.run_id,
            experiment=spec.experiment,
            params=spec.params,
            device=flags.device,
            status="failed",
            mode=mode,
            started_at=started_at,
            finished_at=utc_now(),
            reused=False,
        )
        update_run(
            registry,
            run_id=spec.run_id,
            experiment=spec.experiment,
            run_path=spec.run_dir,
            weights_path=spec.weights_path,
            status="failed",
            metrics={"error": str(exc)},
            mode=mode,
            reused=False,
        )
        print(f"  [fail]  {spec.run_id}: {exc}", file=sys.stderr)
        release_runtime_memory()
        raise


def run_pipeline(flags: PipelineFlags) -> int:
    cfg = load_config(flags.config_path)
    if flags.test:
        cfg = apply_test_mode(cfg)
    mode = cfg.get("mode", "production")

    out_root = output_root(cfg)
    registry_path = out_root / "registry.json"
    registry = load_registry(registry_path, pipeline="pipeline2", mode=mode)
    registry["pipeline"] = "pipeline2"
    registry["mode"] = mode

    if flags.list_runs:
        runs = registry.get("runs", {})
        print(f"=== PIPELINE 2 RUNS ({mode}) — {len(runs)} entries ===")
        for run_id in sorted(runs):
            entry = runs[run_id]
            if flags.experiment and entry.get("experiment") != flags.experiment:
                continue
            print(
                f"{run_id}  [{entry.get('status')}]  "
                f"exp={entry.get('experiment')}  "
                f"F1={entry.get('metrics', {}).get('f1_macro')}"
            )
        return 0

    order = experiment_order(cfg)
    if flags.experiment:
        if flags.experiment not in order:
            raise ValueError(f"unknown experiment: {flags.experiment}")
        run_experiments = [flags.experiment]
    else:
        run_experiments = order

    if flags.dry_run:
        specs = _plan_all_specs(
            cfg,
            experiment_filter=flags.experiment,
            run_id_filter=flags.run_id,
        )
        _print_run_plan(specs, mode)
        return 0

    yolo_source = yolo_best_source(cfg)
    yolo_best = load_pipeline1_best(yolo_source)
    registry["yolo_ref"] = yolo_source.as_posix()
    print(f"YOLO reference: {yolo_best.get('run_id')} ({yolo_best.get('weights')})")

    if flags.test:
        print("=== TEST MODE — full flow, 1 epoch per run ===")
        print(f"Output: {out_root}")
    if not flags.in_process:
        print("Run isolation: subprocess per run (RAM released after each run)")

    if not flags.skip_crops:
        crops_root = crops_dir(cfg)
        if not crops_exist(crops_root):
            print(f"\n=== Preparing crops → {crops_root} ===")
            summary = ensure_crops(
                yolo_yaml=yolo_yaml_path(cfg),
                crops_root=crops_root,
                classes_path=classes_path(cfg),
                padding=float(cfg.get("crops", {}).get("padding", 0.05)),
            )
            if summary is not None:
                print(f"Saved {summary['total_crops_saved']} crops")
        else:
            print(f"Crops dataset ready: {crops_root}")
    else:
        crops_root = crops_dir(cfg)
        if not crops_exist(crops_root):
            raise FileNotFoundError(
                f"crops not found at {crops_root}; remove --skip-crops or build crops first"
            )

    state: dict[str, Any] = {}
    final_best: dict[str, Any] | None = None

    for experiment in run_experiments:
        if experiment != order[0] and not state:
            state = _load_prior_state(out_root, experiment, order)

        print(f"\n=== {experiment} ===")
        param_list = build_runs_for_experiment(cfg, experiment, state)
        exp_specs: list[RunSpec] = []
        for params in param_list:
            run_id = build_run_id(params, pipeline="pipeline2")
            if flags.run_id and run_id != flags.run_id:
                continue
            exp_specs.append(
                RunSpec(
                    experiment=experiment,
                    params=params,
                    run_id=run_id,
                    run_dir=run_directory(out_root, experiment, run_id),
                    weights_path=run_directory(out_root, experiment, run_id)
                    / weights_filename(run_id, pipeline="pipeline2"),
                )
            )

        if not exp_specs:
            continue

        completed_specs: list[dict[str, Any]] = []
        for spec in exp_specs:
            try:
                entry = _execute_run(spec, cfg=cfg, registry=registry, flags=flags)
                completed_specs.append(entry)
                save_registry(registry_path, registry)
            except Exception:
                if flags.run_id:
                    return 1
                continue

        if not completed_specs:
            print(f"  [warn] no completed runs in {experiment}", file=sys.stderr)
            return 1

        metric = cfg["experiments"][experiment].get("metric", "f1_macro")
        run_spec_dicts = [
            {
                "run_id": s["run_id"],
                "experiment": s["experiment"],
                "run_dir": Path(s["dir"]),
                "weights_path": Path(s["weights"]),
                "params": s["hyperparams"],
            }
            for s in completed_specs
        ]
        best_entry = select_experiment_best(run_spec_dicts, metric)
        write_experiment_best(out_root, experiment, best_entry, metric)
        set_experiment_best(registry, experiment, best_entry["run_id"])
        state = state_from_params(best_entry["hyperparams"])
        final_best = best_entry
        save_registry(registry_path, registry)
        print(
            f"  [best]  {experiment} → {best_entry['run_id']} "
            f"({metric}={best_entry['metrics'].get(metric)})"
        )
        release_runtime_memory()

    if final_best is None:
        print("No best model selected.", file=sys.stderr)
        return 1

    set_best_overall(registry, final_best["run_id"])
    save_registry(registry_path, registry)
    best_json = finalize_best(out_root, final_best, yolo_best, yolo_source, mode)
    generate_report(out_root, registry, mode)

    print(f"\n=== PIPELINE 2 COMPLETE ({mode}) ===")
    print(f"Best CNN: {final_best['run_id']}")
    print(f"Weights: {best_json.parent / Path(final_best['weights']).name}")
    print(f"YOLO ref: {yolo_source}")
    print(f"Report: {out_root / 'report' / 'summary.md'}")
    return 0


def parse_args(argv: list[str] | None = None) -> PipelineFlags:
    parser = argparse.ArgumentParser(description="Pipeline 2 — YOLO + CNN hybrid benchmark")
    parser.add_argument("--config", default="configs/pipeline2.yaml", help="Config YAML path")
    parser.add_argument("--test", action="store_true", help="Full flow with 1 epoch per run")
    parser.add_argument("--dry-run", action="store_true", help="Print run plan only")
    parser.add_argument("--force", action="store_true", help="Re-train even if run exists")
    parser.add_argument("--device", default=None, help='Device: "0", "cpu", etc.')
    parser.add_argument("--skip-eval", action="store_true", help="Skip post-train validation metrics")
    parser.add_argument("--skip-crops", action="store_true", help="Skip crop dataset preparation")
    parser.add_argument("--experiment", default=None, help="Run only one experiment")
    parser.add_argument("--run-id", default=None, help="Run only one run_id")
    parser.add_argument("--list-runs", action="store_true", help="List registry entries")
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="Run train/eval in current process (default: subprocess per run for RAM safety)",
    )
    args = parser.parse_args(argv)

    if args.dry_run and args.test:
        parser.error("use either --dry-run or --test, not both")

    return PipelineFlags(
        config_path=Path(args.config),
        test=args.test,
        dry_run=args.dry_run,
        force=args.force,
        device=args.device,
        skip_eval=args.skip_eval,
        skip_crops=args.skip_crops,
        experiment=args.experiment,
        run_id=args.run_id,
        list_runs=args.list_runs,
        in_process=args.in_process,
    )


def main(argv: list[str] | None = None) -> int:
    flags = parse_args(argv)
    return run_pipeline(flags)
