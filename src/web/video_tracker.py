from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MIN_HITS = 2
MAX_MISSES = 6
EMA_ALPHA = 0.72


@dataclass
class _TrackState:
    track_id: int
    bbox: list[float]
    class_name: str
    confidence: float
    hits: int = 1
    misses: int = 0


class VideoTrackStabilizer:
    """Smooth bbox positions and labels across video frames."""

    def __init__(self) -> None:
        self._tracks: dict[int, _TrackState] = {}

    def reset(self) -> None:
        self._tracks.clear()

    @staticmethod
    def _smooth_bbox(previous: list[float], current: list[float]) -> list[float]:
        weight = EMA_ALPHA
        return [
            weight * previous[i] + (1.0 - weight) * current[i]
            for i in range(4)
        ]

    def update(self, observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen_ids: set[int] = set()

        for item in observations:
            track_id = int(item["track_id"])
            seen_ids.add(track_id)
            bbox = [float(v) for v in item["bbox"]]
            class_name = item["class_name"]
            confidence = float(item["confidence"])

            existing = self._tracks.get(track_id)
            if existing is None:
                self._tracks[track_id] = _TrackState(
                    track_id=track_id,
                    bbox=bbox,
                    class_name=class_name,
                    confidence=confidence,
                )
                continue

            existing.bbox = self._smooth_bbox(existing.bbox, bbox)
            existing.hits += 1
            existing.misses = 0

        for track_id, state in list(self._tracks.items()):
            if track_id in seen_ids:
                continue
            state.misses += 1
            if state.misses > MAX_MISSES:
                del self._tracks[track_id]

        visible: list[dict[str, Any]] = []
        for state in self._tracks.values():
            if state.hits < MIN_HITS or state.misses > MAX_MISSES:
                continue
            visible.append(
                {
                    "bbox": [round(v, 2) for v in state.bbox],
                    "class_name": state.class_name,
                    "confidence": round(state.confidence, 6),
                }
            )
        return visible
