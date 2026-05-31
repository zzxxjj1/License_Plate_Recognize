"""Modern FastALPR-based recognition pipeline."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .schema import plate_record, scalar_confidence


DEFAULT_DETECTOR_MODEL = "yolo-v9-t-384-license-plate-end2end"
DEFAULT_OCR_MODEL = "cct-xs-v2-global-model"


class RecognitionError(RuntimeError):
    """Raised when recognition cannot run because of input or dependency problems."""


def recognize_image(
    *,
    image_path: str | Path,
    output_dir: str | Path = "outputs",
    conf_threshold: float = 0.4,
    device: str = "auto",
) -> dict[str, Any]:
    image_path = Path(image_path)
    output_dir = Path(output_dir)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image path is not a file: {image_path}")

    try:
        import cv2
    except ImportError as exc:
        raise RecognitionError(
            "Missing dependency 'opencv-python'. Install dependencies with: "
            "python -m pip install -r requirements.txt"
        ) from exc

    image = cv2.imread(str(image_path))
    if image is None:
        raise RecognitionError(f"Failed to read image: {image_path}")

    timings: dict[str, float] = {}
    total_start = time.perf_counter()

    try:
        from fast_alpr import ALPR
    except ImportError as exc:
        raise RecognitionError(
            "Missing dependency 'fast-alpr'. Install it with: "
            "python -m pip install -r requirements.txt"
        ) from exc

    init_start = time.perf_counter()
    alpr, provider_mode = _build_alpr(ALPR, conf_threshold=conf_threshold, device=device)
    timings["model_init_ms"] = elapsed_ms(init_start)

    inference_start = time.perf_counter()
    alpr_results = alpr.predict(image)
    timings["inference_ms"] = elapsed_ms(inference_start)

    plates = [_serialize_plate(result) for result in alpr_results]

    annotated = _draw_annotations(cv2, image.copy(), plates)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = output_dir / f"{image_path.stem}_annotated.jpg"
    result_path = output_dir / f"{image_path.stem}_result.json"

    annotation_start = time.perf_counter()
    if not cv2.imwrite(str(annotated_path), annotated):
        raise RecognitionError(f"Failed to write annotated image: {annotated_path}")
    timings["annotation_write_ms"] = elapsed_ms(annotation_start)

    result = {
        "input_path": str(image_path),
        "backend": "modern",
        "provider_mode": provider_mode,
        "image_size": {"width": int(image.shape[1]), "height": int(image.shape[0])},
        "timings_ms": {},
        "plates": plates,
        "outputs": {
            "annotated_image": str(annotated_path),
            "result_json": str(result_path),
        },
    }

    timings["total_ms"] = elapsed_ms(total_start)
    result["timings_ms"] = {key: round(value, 3) for key, value in timings.items()}

    json_start = time.perf_counter()
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["timings_ms"]["json_write_ms"] = round(elapsed_ms(json_start), 3)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _build_alpr(ALPR: Any, *, conf_threshold: float, device: str) -> tuple[Any, str]:
    try:
        return (
            ALPR(
                detector_model=DEFAULT_DETECTOR_MODEL,
                detector_conf_thresh=conf_threshold,
                ocr_model=DEFAULT_OCR_MODEL,
                ocr_device=device,
            ),
            device,
        )
    except Exception as exc:
        if device != "auto":
            raise RecognitionError(f"Failed to initialize FastALPR: {exc}") from exc

    try:
        return (
            ALPR(
                detector_model=DEFAULT_DETECTOR_MODEL,
                detector_conf_thresh=conf_threshold,
                detector_providers=["CPUExecutionProvider"],
                ocr_model=DEFAULT_OCR_MODEL,
                ocr_device="cpu",
                ocr_providers=["CPUExecutionProvider"],
            ),
            "cpu-fallback",
        )
    except Exception as cpu_exc:
        raise RecognitionError(f"Failed to initialize FastALPR: {cpu_exc}") from cpu_exc


def _serialize_plate(result: Any) -> dict[str, Any]:
    detection = result.detection
    ocr = result.ocr
    bbox = detection.bounding_box
    bbox_xyxy = [int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)]

    raw_text = ""
    ocr_confidence = None
    if ocr is not None:
        raw_text = getattr(ocr, "text", "") or ""
        ocr_confidence = scalar_confidence(getattr(ocr, "confidence", None))

    return plate_record(
        raw_text=raw_text,
        detection_confidence=scalar_confidence(getattr(detection, "confidence", None)),
        ocr_confidence=ocr_confidence,
        bbox_xyxy=bbox_xyxy,
    )


def _draw_annotations(cv2: Any, image: Any, plates: list[dict[str, Any]]) -> Any:
    for plate in plates:
        x1, y1, x2, y2 = plate["bbox_xyxy"]
        cv2.rectangle(image, (x1, y1), (x2, y2), (36, 255, 12), 2)

        label = plate["normalized_text"] or plate["raw_text"] or "plate"
        confidence = plate["confidence"]
        if confidence is not None:
            label = f"{label} {confidence * 100:.1f}%"

        text_y = max(y1 - 10, 20)
        cv2.putText(
            image,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return image
