from __future__ import annotations

from pathlib import Path
from typing import Any

from src.pipelines.flow1_pipeline import Flow1YoloYoloPipeline
from src.pipelines.flow2_pipeline import Flow2YoloCnnPipeline
from src.web.load_models import BestModelPaths, resolve_best_models


class InferenceService:
    def __init__(self, project_root: Path, device: str | None = None) -> None:
        self.project_root = project_root
        self.device = device
        self.paths = resolve_best_models(project_root)
        self._flow1: Flow1YoloYoloPipeline | None = None
        self._flow2: Flow2YoloCnnPipeline | None = None

    @property
    def flow1(self) -> Flow1YoloYoloPipeline:
        if self._flow1 is None:
            self._flow1 = Flow1YoloYoloPipeline(
                self.paths.yolo_weights,
                device=self.device,
            )
        return self._flow1

    @property
    def flow2(self) -> Flow2YoloCnnPipeline:
        if self._flow2 is None:
            if self.paths.cnn_weights is None:
                raise RuntimeError("Pipeline 2 CNN weights are not available")
            self._flow2 = Flow2YoloCnnPipeline(
                yolo_model_path=self.paths.yolo_weights,
                cnn_checkpoint_path=self.paths.cnn_weights,
                device=self.device,
            )
        return self._flow2

    def model_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "pipeline1": {
                "yolo_run_id": self.paths.yolo_run_id,
                "yolo_weights": str(self.paths.yolo_weights),
                "imgsz": self.paths.yolo_imgsz,
            },
            "pipeline2": {
                "yolo_run_id": self.paths.yolo_run_id,
                "yolo_weights": str(self.paths.yolo_weights),
                "imgsz": self.paths.yolo_imgsz,
            },
        }
        if self.paths.cnn_weights is not None:
            info["pipeline2"]["cnn_run_id"] = self.paths.cnn_run_id
            info["pipeline2"]["cnn_weights"] = str(self.paths.cnn_weights)
        return info
