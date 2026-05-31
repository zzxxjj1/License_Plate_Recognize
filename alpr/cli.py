"""Command-line interface for modern ALPR recognition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .modern import RecognitionError, recognize_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m alpr",
        description="Modern license plate recognition CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    recognize = subparsers.add_parser(
        "recognize",
        help="Detect and recognize license plates in a single image.",
    )
    recognize.add_argument("--image", required=True, help="Path to an input image.")
    recognize.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for JSON and annotated image outputs.",
    )
    recognize.add_argument(
        "--conf-threshold",
        type=float,
        default=0.4,
        help="Detector confidence threshold.",
    )
    recognize.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="OCR device selection.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "recognize":
        return recognize_command(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


def recognize_command(args: argparse.Namespace) -> int:
    try:
        result = recognize_image(
            image_path=Path(args.image),
            output_dir=Path(args.output_dir),
            conf_threshold=args.conf_threshold,
            device=args.device,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except RecognitionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0
