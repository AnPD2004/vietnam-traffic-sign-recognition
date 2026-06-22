from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ClassLabelMapper:
    by_id: dict[int, dict[str, str]]

    @classmethod
    def from_project_root(cls, project_root: Path) -> ClassLabelMapper:
        data_dir = project_root / "data" / "vn-traffic-signs"
        codes = _read_lines(data_dir / "classes.txt")
        names_vie = _read_lines(data_dir / "classes_vie.txt")

        if len(codes) != len(names_vie):
            raise ValueError(
                "classes.txt and classes_vie.txt must have the same number of lines."
            )

        by_id = {
            index: {"code": code, "name_vie": name_vie}
            for index, (code, name_vie) in enumerate(zip(codes, names_vie, strict=True))
        }
        return cls(by_id=by_id)

    def enrich(self, detection: dict[str, Any]) -> dict[str, Any]:
        class_id = detection.get("class_id")
        if class_id is None:
            return detection

        info = self.by_id.get(int(class_id))
        if info is None:
            return detection

        detection["class_code"] = info["code"]
        detection["class_name_vie"] = info["name_vie"]
        return detection

    def enrich_many(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.enrich(detection) for detection in detections]

    def display_name(self, class_id: int) -> str:
        info = self.by_id.get(class_id)
        if info is None:
            return "?"
        return info["name_vie"]


def _read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_image_signs(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    signs: list[dict[str, Any]] = []
    for index, detection in enumerate(detections, start=1):
        signs.append(
            {
                "index": index,
                "class_id": detection.get("class_id"),
                "class_code": detection.get("class_code"),
                "class_name_vie": detection.get("class_name_vie")
                or detection.get("class_name", "?"),
                "confidence": detection.get("confidence", 0.0),
            }
        )
    return signs
