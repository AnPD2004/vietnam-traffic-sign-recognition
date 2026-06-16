"""Execute a single pipeline2 run in an isolated process (RAM/GPU cleanup on exit)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.pipeline2.config import apply_test_mode, load_config
from src.pipeline2.run_one import execute_single_run, read_job_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pipeline 2 — single run worker")
    parser.add_argument("--job", required=True, help="Path to JSON job file")
    args = parser.parse_args(argv)

    job = read_job_file(Path(args.job))
    cfg = load_config(Path(job["config_path"]))
    if job.get("test"):
        cfg = apply_test_mode(cfg)

    try:
        execute_single_run(
            cfg=cfg,
            spec=job["spec"],
            device=job.get("device"),
            skip_eval=bool(job.get("skip_eval")),
        )
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
