"""Result schema helpers for modern license plate recognition."""

from __future__ import annotations

import re
from statistics import mean
from typing import Any


ALNUM_RE = re.compile(r"[^A-Z0-9]")


def normalize_plate_text(text: str | None) -> str:
    """Return an uppercase US-friendly plate string with non-alphanumerics removed."""
    if not text:
        return ""
    return ALNUM_RE.sub("", text.upper())


def scalar_confidence(value: Any) -> float | None:
    """Convert scalar or per-character confidence values into one float."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        values = [float(item) for item in value if item is not None]
        if not values:
            return None
        return float(mean(values))
    return float(value)


def rounded(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def plate_record(
    *,
    raw_text: str,
    detection_confidence: float | None,
    ocr_confidence: float | None,
    bbox_xyxy: list[int],
) -> dict[str, Any]:
    if detection_confidence is not None and ocr_confidence is not None:
        confidence = detection_confidence * ocr_confidence
    else:
        confidence = detection_confidence or ocr_confidence

    return {
        "raw_text": raw_text,
        "normalized_text": normalize_plate_text(raw_text),
        "confidence": rounded(confidence),
        "detection_confidence": rounded(detection_confidence),
        "ocr_confidence": rounded(ocr_confidence),
        "bbox_xyxy": bbox_xyxy,
    }
