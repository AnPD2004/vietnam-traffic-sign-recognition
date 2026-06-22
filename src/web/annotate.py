from __future__ import annotations

import base64
import io
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.web.display_confidence import adjusted_confidence

FONT_CANDIDATES = (
    "arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
)


def load_annotation_font(size: int = 16) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _label_for_detection(det: dict[str, Any], *, pipeline2: bool = False) -> str:
    name = det.get("class_name_vie") or det.get("class_name", "?")
    conf = adjusted_confidence(det.get("confidence", 0.0), pipeline2=pipeline2)
    return f"{name} {conf:.2f}"


def draw_detections_on_image(
    image: Image.Image,
    detections: list[dict[str, Any]],
    box_color: str = "#22c55e",
    text_color: str = "#ffffff",
    *,
    pipeline2: bool = False,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None,
) -> None:
    draw = ImageDraw.Draw(image)
    label_font = font or load_annotation_font()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = _label_for_detection(det, pipeline2=pipeline2)

        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)

        text_bbox = draw.textbbox((x1, y1), label, font=label_font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        text_y = max(0, y1 - text_h - 4)

        draw.rectangle(
            [x1, text_y, x1 + text_w + 6, text_y + text_h + 4],
            fill=box_color,
        )
        draw.text((x1 + 3, text_y + 2), label, fill=text_color, font=label_font)


def draw_detections_on_bgr_frame(
    frame_bgr: np.ndarray,
    detections: list[dict[str, Any]],
    box_color: str = "#22c55e",
    text_color: str = "#ffffff",
    *,
    pipeline2: bool = False,
) -> None:
    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw_detections_on_image(
        image,
        detections,
        box_color=box_color,
        text_color=text_color,
        pipeline2=pipeline2,
    )
    frame_bgr[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def annotate_detections(
    image_path: str,
    detections: list[dict[str, Any]],
    box_color: str = "#22c55e",
    text_color: str = "#ffffff",
    pipeline2: bool = False,
) -> str:
    image = Image.open(image_path).convert("RGB")
    draw_detections_on_image(
        image,
        detections,
        box_color=box_color,
        text_color=text_color,
        pipeline2=pipeline2,
    )

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
