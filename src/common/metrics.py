from __future__ import annotations

from typing import Any


def select_best_run(
    run_entries: list[dict[str, Any]],
    metric: str,
) -> dict[str, Any]:
    if not run_entries:
        raise ValueError("no runs to select from")

    def sort_key(entry: dict[str, Any]) -> tuple[float, float, float]:
        metrics = entry.get("metrics") or {}
        primary = metrics.get(metric)
        if primary is None:
            primary = -1.0
        map50 = metrics.get("map50")
        if map50 is None:
            map50 = -1.0
        fps = metrics.get("fps")
        if fps is None:
            fps = -1.0
        return (float(primary), float(map50), float(fps))

    return max(run_entries, key=sort_key)
