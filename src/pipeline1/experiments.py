from __future__ import annotations

from copy import deepcopy
from typing import Any


def baseline_params(cfg: dict[str, Any]) -> dict[str, Any]:
    baseline = deepcopy(cfg["baseline"])
    return {
        "epochs": int(baseline["epochs"]),
        "imgsz": int(baseline["imgsz"]),
        "lr": float(baseline["lr"]),
        "batch": int(baseline["batch"]),
        "mosaic": int(baseline["mosaic"]),
        "patience": int(baseline["patience"]),
        "seed": int(baseline["seed"]),
        "workers": int(baseline.get("workers", 4)),
    }


def state_from_params(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": params["model"],
        "mosaic": int(params["mosaic"]),
        "imgsz": int(params["imgsz"]),
        "lr": float(params["lr"]),
        "batch": int(params["batch"]),
        "epochs": int(params["epochs"]),
    }


def build_runs_for_experiment(
    cfg: dict[str, Any],
    experiment: str,
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    exp_cfg = cfg["experiments"][experiment]
    base = baseline_params(cfg)

    if experiment == "exp1_model_compare":
        runs: list[dict[str, Any]] = []
        for mosaic in exp_cfg["grid"]["mosaic"]:
            for model in cfg["models"]:
                runs.append({**base, "model": model, "mosaic": int(mosaic)})
        return runs

    if experiment == "exp2_imgsz":
        if not state:
            raise ValueError("exp2_imgsz requires state from exp1_model_compare")
        runs = []
        for imgsz in exp_cfg["grid"]["imgsz"]:
            params = {
                **base,
                "model": state["model"],
                "mosaic": state["mosaic"],
                "imgsz": int(imgsz),
            }
            for key, value in exp_cfg.get("fixed", {}).items():
                params[key] = value
            runs.append(params)
        return runs

    if experiment == "exp3_lr":
        if not state:
            raise ValueError("exp3_lr requires state from previous experiments")
        runs = []
        for lr in exp_cfg["grid"]["lr"]:
            params = {
                **base,
                "model": state["model"],
                "mosaic": state["mosaic"],
                "imgsz": state["imgsz"],
                "lr": float(lr),
            }
            for key, value in exp_cfg.get("fixed", {}).items():
                params[key] = value
            runs.append(params)
        return runs

    if experiment == "exp4_batch":
        if not state:
            raise ValueError("exp4_batch requires state from previous experiments")
        runs = []
        for batch in exp_cfg["grid"]["batch"]:
            params = {
                **base,
                "model": state["model"],
                "mosaic": state["mosaic"],
                "imgsz": state["imgsz"],
                "lr": state["lr"],
                "batch": int(batch),
            }
            for key, value in exp_cfg.get("fixed", {}).items():
                params[key] = value
            runs.append(params)
        return runs

    raise ValueError(f"unknown experiment: {experiment}")
