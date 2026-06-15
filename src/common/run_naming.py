from __future__ import annotations

from pathlib import Path


def fmt_lr(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return f"lr{text}"


def build_run_id(params: dict, pipeline: str = "pipeline1") -> str:
    if pipeline == "pipeline1":
        parts = [
            str(params["model"]),
            f"e{int(params['epochs'])}",
            f"sz{int(params['imgsz'])}",
            fmt_lr(float(params["lr"])),
            f"b{int(params['batch'])}",
            f"mos{int(params['mosaic'])}",
        ]
    elif pipeline == "pipeline2":
        strategy = params.get("strategy", "finetune")
        strat_tag = "frz" if strategy == "frozen" else "ft"
        parts = [
            str(params["model"]),
            f"e{int(params['epochs'])}",
            f"isz{int(params['input_size'])}",
            fmt_lr(float(params["lr"])),
            f"b{int(params['batch'])}",
            strat_tag,
        ]
    else:
        raise ValueError(f"unsupported pipeline: {pipeline}")
    return "_".join(parts)


def weights_filename(run_id: str, pipeline: str = "pipeline1") -> str:
    ext = ".pt" if pipeline == "pipeline1" else ".pth"
    return f"{run_id}{ext}"


def run_directory(output_root: Path, experiment: str, run_id: str) -> Path:
    return output_root / "runs" / experiment / run_id
