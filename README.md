# License Plate Recognition

A modernized Automatic License Plate Recognition (ALPR) project for CSE 5524.
The repository now includes a lightweight command-line recognition pipeline
while preserving the original OpenCV + KNN coursework implementation as a
historical baseline.

## Features

- Modern license plate detection and OCR with FastALPR + ONNX Runtime
- Single-image CLI: `python -m alpr recognize`
- JSON output with text, confidence scores, bounding boxes, and timings
- Annotated image output for visual inspection
- CPU fallback when the default ONNX provider cannot initialize
- Legacy OpenCV/KNN baseline kept for comparison

## How It Works

The modern pipeline uses two stages:

1. Detect the license plate region with `yolo-v9-t-384-license-plate-end2end`
2. Recognize the plate text with `cct-xs-v2-global-model`

Both models are loaded through [FastALPR](https://github.com/ankandrew/fast-alpr).
Model files are not committed to this repository. The first run may download
them into your local model cache.

The old pipeline is still available in `Plate.py`, `Segment.py`, `Predict.py`,
and `main.py`. It uses edge detection, contour filtering, manual character
segmentation, and KNN classification.

## Installation

Use Python 3.10 or newer. On Apple Silicon macOS, Python 3.12 via Homebrew is
recommended:

```bash
brew install python@3.12
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If you already have Python 3.10+ available:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Quick Start

Run the modern recognizer on a sample image:

```bash
source .venv/bin/activate
python -m alpr recognize --image images/Cars1.png --output-dir outputs
```

Run it on your own image:

```bash
python -m alpr recognize --image /path/to/car.jpg --output-dir outputs
```

Optional arguments:

```bash
python -m alpr recognize \
  --image images/Cars1.png \
  --output-dir outputs \
  --conf-threshold 0.4 \
  --device auto
```

`--device` accepts `auto`, `cpu`, or `cuda`. In `auto` mode, the CLI first lets
ONNX Runtime choose the available provider. If that initialization fails, it
falls back to CPU execution instead of crashing.

## Outputs

Each run prints the JSON result and writes two files:

- `outputs/<input_stem>_result.json`
- `outputs/<input_stem>_annotated.jpg`

Example result:

```json
{
  "input_path": "images/Cars1.png",
  "backend": "modern",
  "provider_mode": "auto",
  "image_size": {
    "width": 400,
    "height": 300
  },
  "timings_ms": {
    "model_init_ms": 120.392,
    "inference_ms": 38.348,
    "annotation_write_ms": 2.028,
    "total_ms": 1125.821,
    "json_write_ms": 0.287
  },
  "plates": [
    {
      "raw_text": "CRA1G",
      "normalized_text": "CRA1G",
      "confidence": 0.861038,
      "detection_confidence": 0.906435,
      "ocr_confidence": 0.949917,
      "bbox_xyxy": [145, 205, 283, 240]
    }
  ],
  "outputs": {
    "annotated_image": "outputs/Cars1_annotated.jpg",
    "result_json": "outputs/Cars1_result.json"
  }
}
```

`normalized_text` is uppercased and stripped to `A-Z` and `0-9`, which is a
simple US-friendly default. The project does not enforce state-specific plate
formats.

If no plate is detected, the JSON still contains `plates: []`, and the annotated
image is saved without boxes.

## Example

On a test image with a New York plate reading `FEC-2371`, the modern system
returned:

```text
FEC2371
```

The hyphen is removed by normalization.

## Performance Notes

First runs are slower because model initialization and model downloads dominate
startup time. On a verified local run with `images/Cars1.png`, the first run
measured `36.355 ms` inference and `6677.383 ms` total wall time.

After models were cached, a CPU-fallback run measured `25.702 ms` inference and
`234.117 ms` total wall time. In a non-sandbox Terminal run on this machine,
`provider_mode` stayed as `auto`, meaning the CoreML-capable provider stack
initialized successfully.

## Legacy Baseline

The original coursework system can still be run:

```bash
python main.py
```

It is useful for comparing a classical computer vision pipeline against the
modern detector + OCR approach, but the recommended entry point is now:

```bash
python -m alpr recognize --image images/Cars1.png --output-dir outputs
```

## Project Structure

```text
alpr/
  cli.py       # command-line interface
  modern.py    # FastALPR pipeline and output writing
  schema.py    # result formatting and text normalization

Plate.py       # legacy plate localization
Segment.py     # legacy character segmentation
Predict.py     # legacy KNN classifier
main.py        # legacy demo entry point
```

## Troubleshooting

If `python3` is still macOS system Python 3.9, use the project virtual
environment:

```bash
source .venv/bin/activate
python --version
```

If ONNX Runtime cannot initialize the default provider, the CLI should fall
back to CPU and report:

```json
"provider_mode": "cpu-fallback"
```

This is expected in restricted environments. In a normal Terminal session on
this machine, the same command ran with:

```json
"provider_mode": "auto"
```
