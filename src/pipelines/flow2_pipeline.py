from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.cnn_model import CnnClassifier
from src.common.image_ops import crop_bbox


class Flow2YoloCnnPipeline:
    """Flow 2: YOLO detect + CNN classify."""

    FLOW_ID = "flow2_yolo_cnn"

    def __init__(
        self,
        yolo_model_path: str | Path,
        cnn_checkpoint_path: str | Path,
        device: str | None = None,
    ) -> None:
        from ultralytics import YOLO

        self.yolo_model_path = Path(yolo_model_path)
        if not self.yolo_model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {self.yolo_model_path}")

        self.cnn_checkpoint_path = Path(cnn_checkpoint_path)
        if not self.cnn_checkpoint_path.exists():
            raise FileNotFoundError(f"CNN checkpoint not found: {self.cnn_checkpoint_path}")

        self.device = device
        self.yolo_model = YOLO(str(self.yolo_model_path))
        self.cnn_classifier = CnnClassifier(self.cnn_checkpoint_path, device=device)

    def predict(
        self,
        source: str | Path,
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 640,
        max_det: int = 300,
        crop_padding: float = 0.05,
        save: bool = False,
        save_txt: bool = False,
        project: str | Path | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        results = self.yolo_model.predict(
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
        return self._to_contract(results, crop_padding=crop_padding)

    def _to_contract(self, results: list[Any], crop_padding: float) -> dict[str, Any]:
        detections: list[dict[str, Any]] = []

        for result in results:
            if result.boxes is None or result.orig_img is None:
                continue

            boxes_xyxy = result.boxes.xyxy.cpu().tolist()
            detector_confidences = result.boxes.conf.cpu().tolist()

            for bbox, detector_confidence in zip(
                boxes_xyxy, detector_confidences, strict=True
            ):
                crop = crop_bbox(result.orig_img, bbox, padding=crop_padding)
                class_id, _, confidence = self.cnn_classifier.predict_crop(crop)
                class_name = self.yolo_model.names[class_id]

                detections.append(
                    {
                        "bbox": [round(v, 2) for v in bbox],
                        "detector_confidence": round(float(detector_confidence), 6),
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": round(confidence, 6),
                        "source": {"bbox": "yolo", "class": "cnn"},
                    }
                )

        return {"flow": self.FLOW_ID, "detections": detections}
