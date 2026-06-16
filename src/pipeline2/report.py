from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def generate_report(output_root: Path, registry: dict[str, Any], mode: str) -> None:
    report_dir = output_root / "report"
    by_experiment_dir = report_dir / "by_experiment"
    report_dir.mkdir(parents=True, exist_ok=True)
    by_experiment_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Pipeline 2 Report ({mode})",
        "",
        f"Total runs: {registry.get('total_runs', 0)}",
        f"Best overall: `{registry.get('best_overall')}`",
        "",
        "## Best per experiment",
        "",
    ]

    runs = registry.get("runs", {})
    for experiment, run_id in registry.get("best_per_experiment", {}).items():
        entry = runs.get(run_id, {})
        metric = entry.get("metrics", {}).get("f1_macro")
        lines.append(f"- **{experiment}**: `{run_id}` (F1-macro={metric})")
        _write_experiment_report(by_experiment_dir, experiment, runs, run_id)

    lines.extend(["", "## All runs", ""])
    for run_id in sorted(runs):
        entry = runs[run_id]
        status = entry.get("status")
        reused = entry.get("reused", False)
        metric = entry.get("metrics", {}).get("f1_macro")
        suffix = " [reused]" if reused else ""
        lines.append(f"- `{run_id}` — {status}, F1-macro={metric}{suffix}")

    (report_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (report_dir / "summary.json").write_text(
        json.dumps(
            {
                "mode": mode,
                "best_overall": registry.get("best_overall"),
                "best_per_experiment": registry.get("best_per_experiment"),
                "runs": runs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_experiment_report(
    report_dir: Path,
    experiment: str,
    runs: dict[str, Any],
    best_run_id: str,
) -> None:
    experiment_runs = [
        (run_id, entry)
        for run_id, entry in runs.items()
        if entry.get("experiment") == experiment
    ]
    lines = [
        f"# {experiment}",
        "",
        f"Best: `{best_run_id}`",
        "",
        "| run_id | F1-macro | accuracy | status |",
        "|--------|----------|----------|--------|",
    ]
    for run_id, entry in sorted(experiment_runs):
        metrics = entry.get("metrics", {})
        lines.append(
            f"| `{run_id}` | {metrics.get('f1_macro')} | "
            f"{metrics.get('accuracy')} | {entry.get('status')} |"
        )
    (report_dir / f"{experiment}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
