from __future__ import annotations

PIPELINE2_CONFIDENCE_OFFSET = 0.012


def adjusted_confidence(confidence: float, *, pipeline2: bool = False) -> float:
    value = float(confidence)
    if pipeline2:
        return max(0.0, value - PIPELINE2_CONFIDENCE_OFFSET)
    return value
