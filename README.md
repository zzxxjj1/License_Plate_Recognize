# License Plate Recognition

Final project for CSE 5524, now upgraded with a modern Automatic License Plate
Recognition (ALPR) command-line demo.

The original OpenCV + KNN implementation is still included as a historical
baseline in `Plate.py`, `Segment.py`, `Predict.py`, and `main.py`. The default
workflow is now the modern pipeline in the `alpr` package.

## Modern Pipeline

The modern CLI uses [FastALPR](https://github.com/ankandrew/fast-alpr), which
combines a lightweight ONNX license plate detector with a plate-specific OCR
model:

- detector: `yolo-v9-t-384-license-plate-end2end`
- OCR: `cct-xs-v2-global-model`
- backend: `modern`

Model weights are not stored in this repository. The first run may download
models into your local package/model cache.

## Installation

Use Python 3.10 or newer.

```bash
brew install python@3.12
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Usage

Run recognition on one image:

```bash
python -m alpr recognize --image images/Cars1.png --output-dir outputs
```

Optional arguments:

```bash
python -m alpr recognize \
  --image images/Cars1.png \
  --output-dir outputs \
  --conf-threshold 0.4 \
  --device auto
```

`--device` can be `auto`, `cpu`, or `cuda`.

In `auto` mode, the CLI will fall back to CPU execution if the default ONNX
provider cannot initialize on the current machine.

## Outputs

Each run prints the JSON result and writes:

- `outputs/<input_stem>_result.json`
- `outputs/<input_stem>_annotated.jpg`

If no plate is detected, the JSON is still written with `plates: []`, and the
annotated image is saved without boxes.

Example JSON shape:

```json
{
  "input_path": "images/Cars1.png",
  "backend": "modern",
  "provider_mode": "auto",
  "image_size": {
    "width": 1024,
    "height": 768
  },
  "timings_ms": {
    "model_init_ms": 120.0,
    "inference_ms": 35.0,
    "annotation_write_ms": 4.0,
    "total_ms": 160.0,
    "json_write_ms": 1.0
  },
  "plates": [
    {
      "raw_text": "ABC123",
      "normalized_text": "ABC123",
      "confidence": 0.91,
      "detection_confidence": 0.97,
      "ocr_confidence": 0.94,
      "bbox_xyxy": [120, 350, 280, 400]
    }
  ],
  "outputs": {
    "annotated_image": "outputs/Cars1_annotated.jpg",
    "result_json": "outputs/Cars1_result.json"
  }
}
```

`normalized_text` is uppercased and stripped to `A-Z` and `0-9` for a
US-friendly default. The project does not enforce state-specific plate formats.

On the first verified local run with `images/Cars1.png`, model initialization
and first-time downloads dominated startup time. The measured inference step was
`36.355 ms`, with `6677.383 ms` total wall time for the first run. Subsequent
runs avoid the initial model download; a cached CPU-fallback run measured
`25.702 ms` inference and `234.117 ms` total wall time.

## Legacy Baseline

The old coursework pipeline can still be inspected and run separately:

```bash
python main.py
```

It uses OpenCV preprocessing, contour filtering, manual character segmentation,
and KNN classification. It is kept for comparison with the modern detector +
OCR pipeline, not as the recommended path.
