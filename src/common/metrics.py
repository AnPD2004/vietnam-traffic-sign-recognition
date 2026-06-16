from __future__ import annotations

from typing import Any


_MINIMIZE_METRICS = frozenset({"latency_ms_per_image", "inference_ms"})

_TIE_BREAKERS: dict[str, tuple[str, ...]] = {
    "map50_95": ("map50", "fps"),
    "map50": ("fps",),
    "f1_macro": ("accuracy", "latency_ms_per_image"),
}


def _metric_sort_value(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    if value is None:
        return float("inf") if key in _MINIMIZE_METRICS else -1.0
    numeric = float(value)
    if key in _MINIMIZE_METRICS:
        return -numeric
    return numeric


def select_best_run(
    run_entries: list[dict[str, Any]],
    metric: str,
    *,
    tie_breakers: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if not run_entries:
        raise ValueError("no runs to select from")

    resolved_tie_breakers = tie_breakers or _TIE_BREAKERS.get(metric, ())

    def sort_key(entry: dict[str, Any]) -> tuple[float, ...]:
        metrics = entry.get("metrics") or {}
        keys = (metric, *resolved_tie_breakers)
        return tuple(_metric_sort_value(metrics, key) for key in keys)

    return max(run_entries, key=sort_key)
