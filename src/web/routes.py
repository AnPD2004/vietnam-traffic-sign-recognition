from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from flask import Blueprint, current_app, jsonify, request
from werkzeug.utils import secure_filename

from src.web.annotate import annotate_detections

api_bp = Blueprint("api", __name__)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _save_upload() -> Path:
    if "image" not in request.files:
        raise ValueError("Missing image file in form field 'image'")

    file = request.files["image"]
    if not file or not file.filename:
        raise ValueError("No image selected")

    filename = secure_filename(file.filename)
    if not _allowed_file(filename):
        raise ValueError("Unsupported image format. Use JPG, PNG, WEBP, or BMP.")

    upload_dir = Path(current_app.config["UPLOAD_DIR"])
    upload_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(filename).suffix.lower()
    saved_path = upload_dir / f"{uuid.uuid4().hex}{suffix}"
    file.save(saved_path)
    return saved_path


def _build_metrics(elapsed_s: float, detections: list[dict[str, Any]]) -> dict[str, Any]:
    inference_ms = round(elapsed_s * 1000, 2)
    fps = round(1.0 / elapsed_s, 2) if elapsed_s > 0 else 0.0
    return {
        "sign_count": len(detections),
        "inference_ms": inference_ms,
        "fps": fps,
    }


def _predict_response(
    result: dict[str, Any],
    image_path: Path,
    elapsed_s: float,
    box_color: str,
) -> dict[str, Any]:
    detections = result.get("detections", [])
    metrics = _build_metrics(elapsed_s, detections)
    annotated_b64 = annotate_detections(str(image_path), detections, box_color=box_color)

    return {
        "flow": result.get("flow"),
        "detections": detections,
        "metrics": metrics,
        "image_base64": annotated_b64,
    }


@api_bp.get("/health")
def health() -> Any:
    service = current_app.extensions["inference_service"]
    return jsonify({"status": "ok", "models": service.model_info()})


@api_bp.post("/predict/flow1")
def predict_flow1() -> Any:
    service = current_app.extensions["inference_service"]
    image_path: Path | None = None

    try:
        image_path = _save_upload()
        start = time.perf_counter()
        result = service.flow1.predict(
            source=image_path,
            imgsz=service.paths.yolo_imgsz,
        )
        elapsed = time.perf_counter() - start

        payload = _predict_response(result, image_path, elapsed, box_color="#3b82f6")
        payload["pipeline"] = "pipeline1"
        payload["models"] = {
            "yolo": str(service.paths.yolo_weights),
            "cnn": None,
        }
        return jsonify(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        if image_path and image_path.exists():
            image_path.unlink(missing_ok=True)


@api_bp.post("/predict/flow2")
def predict_flow2() -> Any:
    service = current_app.extensions["inference_service"]
    image_path: Path | None = None

    try:
        image_path = _save_upload()
        start = time.perf_counter()
        result = service.flow2.predict(
            source=image_path,
            imgsz=service.paths.yolo_imgsz,
        )
        elapsed = time.perf_counter() - start

        payload = _predict_response(result, image_path, elapsed, box_color="#22c55e")
        payload["pipeline"] = "pipeline2"
        payload["models"] = {
            "yolo": str(service.paths.yolo_weights),
            "cnn": str(service.paths.cnn_weights) if service.paths.cnn_weights else None,
        }
        return jsonify(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        if image_path and image_path.exists():
            image_path.unlink(missing_ok=True)
