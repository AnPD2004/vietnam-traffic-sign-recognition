from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from flask import Blueprint, current_app, jsonify, request, send_file
from werkzeug.utils import secure_filename

from src.web.annotate import annotate_detections
from src.web.class_labels import build_image_signs
from src.web.video_infer import process_video_flow1, process_video_flow2

api_bp = Blueprint("api", __name__)

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".webm", ".avi", ".mov", ".mkv"}


def _allowed_image(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_IMAGE_EXTENSIONS


def _allowed_video(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_VIDEO_EXTENSIONS


def _save_upload(field_name: str, allowed_check, error_message: str) -> Path:
    if field_name not in request.files:
        raise ValueError(f"Missing file in form field '{field_name}'")

    file = request.files[field_name]
    if not file or not file.filename:
        raise ValueError("No file selected")

    filename = secure_filename(file.filename)
    if not allowed_check(filename):
        raise ValueError(error_message)

    upload_dir = Path(current_app.config["UPLOAD_DIR"])
    upload_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(filename).suffix.lower()
    saved_path = upload_dir / f"{uuid.uuid4().hex}{suffix}"
    file.save(saved_path)
    return saved_path


def _save_image_upload() -> Path:
    return _save_upload(
        "image",
        _allowed_image,
        "Unsupported image format. Use JPG, PNG, WEBP, or BMP.",
    )


def _save_video_upload() -> Path:
    return _save_upload(
        "video",
        _allowed_video,
        "Unsupported video format. Use MP4, WEBM, AVI, MOV, or MKV.",
    )


def _build_metrics(elapsed_s: float, sign_count: int) -> dict[str, Any]:
    return {
        "sign_count": sign_count,
        "inference_ms": round(elapsed_s * 1000, 2),
    }


def _predict_response(
    service: Any,
    result: dict[str, Any],
    image_path: Path,
    elapsed_s: float,
    box_color: str,
) -> dict[str, Any]:
    detections = service.label_mapper.enrich_many(result.get("detections", []))
    signs = build_image_signs(detections)
    metrics = _build_metrics(elapsed_s, len(signs))
    annotated_b64 = annotate_detections(str(image_path), detections, box_color=box_color)

    return {
        "flow": result.get("flow"),
        "detections": detections,
        "signs": signs,
        "metrics": metrics,
        "image_base64": annotated_b64,
    }


@api_bp.get("/health")
def health() -> Any:
    service = current_app.extensions["inference_service"]
    return jsonify({"status": "ok", "models": service.model_info()})


@api_bp.get("/outputs/<path:filename>")
def serve_output(filename: str) -> Any:
    output_dir = Path(current_app.config["OUTPUT_DIR"])
    file_path = output_dir / filename
    if not file_path.is_file():
        return jsonify({"error": "Video not found"}), 404

    mimetype = "video/webm" if file_path.suffix.lower() == ".webm" else "video/mp4"
    return send_file(
        file_path,
        mimetype=mimetype,
        conditional=True,
        download_name=filename,
    )


@api_bp.post("/predict/flow1")
def predict_flow1() -> Any:
    service = current_app.extensions["inference_service"]
    image_path: Path | None = None

    try:
        image_path = _save_image_upload()
        start = time.perf_counter()
        result = service.flow1.predict(
            source=image_path,
            imgsz=service.paths.yolo_imgsz,
        )
        elapsed = time.perf_counter() - start

        payload = _predict_response(service, result, image_path, elapsed, box_color="#3b82f6")
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
        image_path = _save_image_upload()
        start = time.perf_counter()
        result = service.flow2.predict(
            source=image_path,
            imgsz=service.paths.yolo_imgsz,
        )
        elapsed = time.perf_counter() - start

        payload = _predict_response(service, result, image_path, elapsed, box_color="#22c55e")
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


def _video_predict_response(
    result: dict[str, Any],
    output_filename: str,
    pipeline: str,
    models: dict[str, Any],
) -> dict[str, Any]:
    return {
        "flow": result.get("flow"),
        "pipeline": pipeline,
        "models": models,
        "metrics": result.get("metrics"),
        "signs": result.get("signs", []),
        "video_url": f"/outputs/{output_filename}",
    }


@api_bp.post("/predict/video/flow1")
def predict_video_flow1() -> Any:
    service = current_app.extensions["inference_service"]
    video_path: Path | None = None

    try:
        video_path = _save_video_upload()
        output_dir = Path(current_app.config["OUTPUT_DIR"])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = f"{uuid.uuid4().hex}.mp4"
        output_path = output_dir / output_filename

        result = process_video_flow1(
            service,
            video_path,
            output_path,
            imgsz=service.paths.yolo_imgsz,
        )

        payload = _video_predict_response(
            result,
            result["output_path"].name,
            pipeline="pipeline1",
            models={
                "yolo": str(service.paths.yolo_weights),
                "cnn": None,
            },
        )
        return jsonify(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        if video_path and video_path.exists():
            video_path.unlink(missing_ok=True)


@api_bp.post("/predict/video/flow2")
def predict_video_flow2() -> Any:
    service = current_app.extensions["inference_service"]
    video_path: Path | None = None

    try:
        video_path = _save_video_upload()
        output_dir = Path(current_app.config["OUTPUT_DIR"])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = f"{uuid.uuid4().hex}.mp4"
        output_path = output_dir / output_filename

        result = process_video_flow2(
            service,
            video_path,
            output_path,
            imgsz=service.paths.yolo_imgsz,
        )

        payload = _video_predict_response(
            result,
            result["output_path"].name,
            pipeline="pipeline2",
            models={
                "yolo": str(service.paths.yolo_weights),
                "cnn": str(service.paths.cnn_weights) if service.paths.cnn_weights else None,
            },
        )
        return jsonify(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        if video_path and video_path.exists():
            video_path.unlink(missing_ok=True)
