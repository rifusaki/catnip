#!/usr/bin/env python3
"""
Thin CLI wrapper around :mod:`src.preprocess.coco_parser`.

Converts the AnimeHeadsv3 COCO-format dataset to flat YOLO format for Stage 1
training (3 classes: body=0, head=1, face=2).

AnimeHeadsv3 ships two category IDs (0, 1) both labelled ``"head"`` and uses
COCO ``bbox`` format ``[x, y, width, height]`` in pixel coordinates.  The
library remaps both categories → class 1 (head), normalises the boxes to YOLO
``(x_center, y_center, w, h)``, and flattens all splits into a single output.

The wrapper handles CLI argument parsing, input validation (translating
:class:`CocoValidationError` into a non-zero exit), output directory
creation, and summary printing.

Usage::

    python scripts/unify/coco_heads.py

    python scripts/unify/coco_heads.py \\
        --input-dir /Volumes/rifuSSD/catnip-data/data/AnimeHeadsv3 \\
        --output-dir /Volumes/rifuSSD/catnip-data/data/staging/ah_coco
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow running the script directly from scripts/unify/ without an installed package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import settings
from src.preprocess.coco_parser import (
    SPLITS,
    CocoValidationError,
    convert_split,
    validate_input_dir,
)

logger = logging.getLogger("unify_coco_heads")


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(
    overall: dict[str, int],
    split_details: dict[str, dict[str, int]],
    output_dir: Path,
) -> None:
    """Print a human-readable conversion summary."""
    header = " AnimeHeadsv3 COCO → YOLO Conversion Summary "
    print(f"\n{'=' * 62}")
    print(f"{header:=^62}")
    print(f"{'=' * 62}")

    for split in SPLITS:
        d = split_details.get(split, {})
        print(f"\n  [{split}]")
        print(f"    Images copied:     {d.get('images_copied', 0):>6}")
        print(f"    Annotations written:{d.get('labels_written', 0):>6}")
        print(f"    Images skipped:    {d.get('images_skipped', 0):>6}  (no annotations / missing file)")
        if d.get("bboxes_dropped", 0):
            print(f"    Bboxes dropped:    {d.get('bboxes_dropped', 0):>6}  (degenerate / out of bounds)")
        if d.get("bboxes_clamped", 0):
            print(f"    Bboxes clamped:    {d.get('bboxes_clamped', 0):>6}  (partial overflow)")

    print(f"\n  [Totals]")
    print(f"    Images copied:     {overall['images_copied']:>6}")
    print(f"    Annotations written:{overall['labels_written']:>6}")
    print(f"    Images skipped:    {overall['images_skipped']:>6}")
    if overall.get("bboxes_dropped", 0):
        print(f"    Bboxes dropped:    {overall['bboxes_dropped']:>6}")
    if overall.get("bboxes_clamped", 0):
        print(f"    Bboxes clamped:    {overall['bboxes_clamped']:>6}  (partial overflow)")

    print(f"\n  Output: {output_dir}")
    print(f"{'=' * 62}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert AnimeHeadsv3 COCO dataset to flat YOLO format for Stage 1.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=settings.paths.ah_coco_dir,
        help="Root directory of the AnimeHeadsv3 dataset (default from config).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.paths.data / "data" / "staging" / "ah_coco",
        help="Output directory. Images go to <output>/images/, labels to <output>/labels/.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )

    args = parser.parse_args()

    # --- Setup logging ---
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(message)s",
    )

    logger.info("Input:  %s", args.input_dir)
    logger.info("Output: %s", args.output_dir)

    # --- Validate input (Law 4 — Fail Fast) ---
    try:
        splits = validate_input_dir(args.input_dir)
    except CocoValidationError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    # --- Create output directories ---
    images_dst = args.output_dir / "images"
    labels_dst = args.output_dir / "labels"
    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    # --- Convert each split ---
    overall: dict[str, int] = {
        "images_copied": 0,
        "labels_written": 0,
        "images_skipped": 0,
        "bboxes_dropped": 0,
        "bboxes_clamped": 0,
    }
    split_details: dict[str, dict[str, int]] = {}

    for split_name in SPLITS:
        logger.info("Processing %s split...", split_name)
        try:
            details = convert_split(
                split_name,
                splits[split_name],
                images_dst,
                labels_dst,
            )
        except (json.JSONDecodeError, TypeError) as exc:
            logger.error("Failed to process %s split: %s", split_name, exc)
            sys.exit(1)

        split_details[split_name] = details
        overall["images_copied"] += details["images_copied"]
        overall["labels_written"] += details["labels_written"]
        overall["images_skipped"] += details["images_skipped"]
        overall["bboxes_dropped"] += details["bboxes_dropped"]
        overall["bboxes_clamped"] += details["bboxes_clamped"]

    # --- Summary ---
    _print_summary(overall, split_details, args.output_dir)


if __name__ == "__main__":
    main()
