from __future__ import annotations

import gc
import shutil
from pathlib import Path
from typing import Any


def release_runtime_memory(*models: Any) -> None:
    """Drop model references and return RAM/GPU memory to the OS when possible."""
    for obj in models:
        try:
            del obj
        except Exception:
            pass

    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


def purge_ultralytics_run_dir(ultra_dir: Path) -> None:
    """Remove Ultralytics training output (often under runs/detect/)."""
    if not ultra_dir.exists():
        return

    shutil.rmtree(ultra_dir, ignore_errors=True)

    parent = ultra_dir.parent
    while parent.parts:
        if parent.name in {"detect", "runs"}:
            break
        if not parent.exists():
            break
        try:
            next(parent.iterdir())
        except StopIteration:
            parent.rmdir()
            parent = parent.parent
            continue
        except OSError:
            break
        break
