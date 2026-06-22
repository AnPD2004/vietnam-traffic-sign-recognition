from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2

from src.common.image_ops import crop_bbox
from src.web.models import InferenceService
from src.web.video_encode import transcode_to_browser_mp4
from src.web.video_tracker import VideoTrackStabilizer

MAX_FRAMES = 300
TRACKER_CONFIG = "bytetrack.yaml"


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    red = int(value[0:2], 16)
    green = int(value[2:4], 16)
    blue = int(value[4:6], 16)
    return blue, green, red


def _label_for_detection(det: dict[str, Any]) -> str:
    name = det.get("class_name", "?")
    conf = det.get("confidence", 0.0)
    return f"{name} {conf:.2f}"


def _draw_detections(
    frame: Any,
    detections: list[dict[str, Any]],
    box_color: str,
) -> None:
    color = _hex_to_bgr(box_color)
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        label = _label_for_detection(det)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )


def _track_yolo_frame(
    model: Any,
    frame: Any,
    imgsz: int,
    device: str | None,
) -> list[Any]:
    return model.track(
        frame,
        imgsz=imgsz,
        conf=0.3,
        iou=0.7,
        device=device,
        persist=True,
        tracker=TRACKER_CONFIG,
        verbose=False,
    )


def _flow1_frame_observations(
    service: InferenceService,
    frame: Any,
    imgsz: int,
) -> list[dict[str, Any]]:
    results = _track_yolo_frame(service.flow1.model, frame, imgsz, service.device)
    observations: list[dict[str, Any]] = []

    for result in results:
        names = result.names
        if result.boxes is None or result.boxes.id is None:
            continue

        boxes_xyxy = result.boxes.xyxy.cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()
        class_ids = result.boxes.cls.cpu().tolist()
        track_ids = result.boxes.id.int().cpu().tolist()

        for bbox, confidence, class_id, track_id in zip(
            boxes_xyxy, confidences, class_ids, track_ids, strict=True
        ):
            class_index = int(class_id)
            observations.append(
                {
                    "track_id": int(track_id),
                    "bbox": bbox,
                    "class_name": names[class_index],
                    "confidence": float(confidence),
                }
            )

    return observations


def _flow2_frame_observations(
    service: InferenceService,
    frame: Any,
    imgsz: int,
    class_cache: dict[int, tuple[str, float]],
    crop_padding: float = 0.05,
) -> list[dict[str, Any]]:
    results = _track_yolo_frame(service.flow2.yolo_model, frame, imgsz, service.device)
    observations: list[dict[str, Any]] = []

    for result in results:
        if result.boxes is None or result.boxes.id is None or result.orig_img is None:
            continue

        boxes_xyxy = result.boxes.xyxy.cpu().tolist()
        track_ids = result.boxes.id.int().cpu().tolist()

        for bbox, track_id in zip(boxes_xyxy, track_ids, strict=True):
            track_id = int(track_id)
            if track_id not in class_cache:
                crop = crop_bbox(result.orig_img, bbox, padding=crop_padding)
                class_id, _, confidence = service.flow2.cnn_classifier.predict_crop(crop)
                class_name = service.flow2.yolo_model.names[class_id]
                class_cache[track_id] = (class_name, float(confidence))

            class_name, confidence = class_cache[track_id]
            observations.append(
                {
                    "track_id": track_id,
                    "bbox": bbox,
                    "class_name": class_name,
                    "confidence": confidence,
                }
            )

    return observations


def _reset_yolo_tracker(model: Any) -> None:
    if hasattr(model, "predictor") and model.predictor is not None:
        model.predictor = None


def _build_video_metrics(
    elapsed_s: float,
    frame_count: int,
    max_signs_per_frame: int,
    truncated: bool,
) -> dict[str, Any]:
    inference_ms = round(elapsed_s * 1000, 2)
    fps = round(frame_count / elapsed_s, 2) if elapsed_s > 0 else 0.0
    return {
        "frame_count": frame_count,
        "sign_count": max_signs_per_frame,
        "inference_ms": inference_ms,
        "fps": fps,
        "truncated": truncated,
    }


def _process_video(
    service: InferenceService,
    video_path: Path,
    output_path: Path,
    imgsz: int,
    box_color: str,
    flow: str,
) -> dict[str, Any]:
    capture: Any = None
    writer: Any = None
    raw_path = output_path.with_suffix(".raw.avi")
    final_path = output_path.with_suffix(".mp4")
    stabilizer = VideoTrackStabilizer()
    class_cache: dict[int, tuple[str, float]] = {}

    try:
        if flow == "flow1":
            _reset_yolo_tracker(service.flow1.model)
        else:
            _reset_yolo_tracker(service.flow2.yolo_model)

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError("Không thể đọc file video.")

        fps = max(1.0, min(float(capture.get(cv2.CAP_PROP_FPS) or 25.0), 60.0))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise ValueError("Video không hợp lệ hoặc không có frame.")

        writer = cv2.VideoWriter(
            str(raw_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise ValueError("Không thể tạo file video đầu ra.")

        frame_count = 0
        max_signs_per_frame = 0
        truncated = False
        start = time.perf_counter()

        while frame_count < MAX_FRAMES:
            ok, frame = capture.read()
            if not ok:
                break

            if flow == "flow1":
                observations = _flow1_frame_observations(service, frame, imgsz)
            else:
                observations = _flow2_frame_observations(
                    service, frame, imgsz, class_cache
                )

            detections = stabilizer.update(observations)
            max_signs_per_frame = max(max_signs_per_frame, len(detections))
            _draw_detections(frame, detections, box_color)
            writer.write(frame)
            frame_count += 1

        if frame_count == MAX_FRAMES:
            truncated = capture.read()[0]

        if frame_count == 0:
            raise ValueError("Video không có frame nào để xử lý.")

        elapsed = time.perf_counter() - start
        metrics = _build_video_metrics(elapsed, frame_count, max_signs_per_frame, truncated)

        writer.release()
        writer = None
        capture.release()
        capture = None

        transcode_to_browser_mp4(raw_path, final_path)
        raw_path.unlink(missing_ok=True)

        return {
            "flow": "flow1_yolo_yolo" if flow == "flow1" else "flow2_yolo_cnn",
            "metrics": metrics,
            "output_path": final_path,
        }
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        raw_path.unlink(missing_ok=True)


def process_video_flow1(
    service: InferenceService,
    video_path: Path,
    output_path: Path,
    imgsz: int,
) -> dict[str, Any]:
    return _process_video(service, video_path, output_path, imgsz, "#3b82f6", "flow1")


def process_video_flow2(
    service: InferenceService,
    video_path: Path,
    output_path: Path,
    imgsz: int,
) -> dict[str, Any]:
    return _process_video(service, video_path, output_path, imgsz, "#22c55e", "flow2")
