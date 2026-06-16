from __future__ import annotations

from copy import deepcopy
from typing import Any


def baseline_params(cfg: dict[str, Any]) -> dict[str, Any]:
    baseline = deepcopy(cfg["baseline"])
    return {
        "epochs": int(baseline["epochs"]),
        "input_size": int(baseline["input_size"]),
        "lr": float(baseline["lr"]),
        "batch": int(baseline["batch"]),
        "strategy": str(baseline.get("strategy", "finetune")),
        "patience": int(baseline["patience"]),
        "seed": int(baseline["seed"]),
        "weight_decay": float(baseline.get("weight_decay", 1e-4)),
        "workers": int(baseline.get("workers", 4)),
    }


def state_from_params(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": params["model"],
        "lr": float(params["lr"]),
        "batch": int(params["batch"]),
        "input_size": int(params["input_size"]),
        "strategy": str(params["strategy"]),
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
        return [{**base, "model": model} for model in cfg["models"]]

    if experiment == "exp2_lr":
        if not state:
            raise ValueError("exp2_lr requires state from exp1_model_compare")
        runs: list[dict[str, Any]] = []
        for lr in exp_cfg["grid"]["lr"]:
            params = {
                **base,
                "model": state["model"],
                "lr": float(lr),
            }
            for key, value in exp_cfg.get("fixed", {}).items():
                params[key] = value
            runs.append(params)
        return runs

    if experiment == "exp3_transfer":
        if not state:
            raise ValueError("exp3_transfer requires state from previous experiments")
        runs = []
        for strategy in exp_cfg["grid"]["strategy"]:
            params = {
                **base,
                "model": state["model"],
                "lr": state["lr"],
                "batch": state["batch"],
                "input_size": state["input_size"],
                "strategy": str(strategy),
            }
            runs.append(params)
        return runs

    raise ValueError(f"unknown experiment: {experiment}")
