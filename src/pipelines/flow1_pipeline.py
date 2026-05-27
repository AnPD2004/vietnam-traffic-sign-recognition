from __future__ import annotations

from pathlib import Path
from typing import Any


class Flow1YoloYoloPipeline:
    """Flow 1: YOLO detect + YOLO classify."""

    FLOW_ID = "flow1_yolo_yolo"

    def __init__(self, model_path: str | Path, device: str | None = None) -> None:
        from ultralytics import YOLO

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        self.device = device

    def predict(
        self,
        source: str | Path,
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 640,
        max_det: int = 300,
        save: bool = False,
        save_txt: bool = False,
        project: str | Path | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        results = self.model.predict(
            source=str(source),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            device=self.device,
            save=save,
            save_txt=save_txt,
            project=str(project) if project else None,
            name=name,
            verbose=False,
        )
        return self._to_contract(results)

    def _to_contract(self, results: list[Any]) -> dict[str, Any]:
        detections: list[dict[str, Any]] = []

        for result in results:
            names = result.names
            if result.boxes is None:
                continue

            boxes_xyxy = result.boxes.xyxy.cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()
            class_ids = result.boxes.cls.cpu().tolist()

            for bbox, confidence, class_id in zip(
                boxes_xyxy, confidences, class_ids, strict=True
            ):
                class_index = int(class_id)
                detections.append(
                    {
                        "bbox": [round(v, 2) for v in bbox],
                        "class_id": class_index,
                        "class_name": names[class_index],
                        "confidence": round(float(confidence), 6),
                        "source": "yolo",
                    }
                )

        return {"flow": self.FLOW_ID, "detections": detections}

