#!/usr/bin/env python3
"""
Thin CLI wrapper around :mod:`src.preprocess.yolo_remap`.

Converts YOLOv8-format head detection datasets into the unified Stage 1 format.

Reads datasets from ``catnip-data/data/anime_head_detection/`` (v1, v2, ani_face),
remaps class 0 → 1 (head), and writes flat image/label directories with
source-specific filename prefixes.

Output is a single flat directory (no train/valid/test split) — splitting is
handled later by ``scripts/unify/stage1.py``.

The wrapper handles CLI argument parsing, output directory resolution,
per-dataset logging, and the grand-total summary.

Usage::

    python scripts/unify/yolo_heads.py
    python scripts/unify/yolo_heads.py --output-dir /custom/output
    python scripts/unify/yolo_heads.py --datasets /Volumes/rifuSSD/catnip-data/data/anime_head_detection/v1
    python scripts/unify/yolo_heads.py --verbose
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running the script directly from scripts/unify/ without an installed package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import settings
from src.preprocess.yolo_remap import (
    DEFAULT_OUTPUT,
    DEFAULT_VARIANTS,
    SPLITS,
    ensure_writable,
    process_dataset,
    resolve_variant,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unify YOLO head-detection datasets into Stage 1 format.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        metavar="PATH",
        help=(
            "Dataset directories relative to project root. "
            "Defaults to all three anime head detection variants "
            "(v1, v2, ani_face_detection)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT,
        help="Output directory relative to project root (default: %(default)s).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    args = parser.parse_args()

    # --- Setup logging ---
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(message)s",
    )

    # --- Resolve dataset specs ---
    project_root = Path(__file__).resolve().parents[2]
    data_root = settings.paths.data

    if args.datasets is None:
        ah_parent = data_root / "data" / "anime_head_detection"
        dataset_specs: list[tuple[Path, str]] = [
            (ah_parent / variant, prefix)
            for variant, prefix in DEFAULT_VARIANTS
        ]
    else:
        dataset_specs = [resolve_variant(p, project_root) for p in args.datasets]

    if not dataset_specs:
        logger.error(
            "No datasets to process. Use --datasets PATH [PATH ...] or omit for defaults."
        )
        sys.exit(1)

    # --- Resolve output directory ---
    if args.output_dir == DEFAULT_OUTPUT:
        output_dir = (data_root / "data" / "staging" / "ah_yolo").resolve()
    else:
        output_dir = (project_root / args.output_dir).resolve()
    try:
        ensure_writable(output_dir)
    except (PermissionError, OSError) as exc:
        logger.error("Cannot create output directory %s: %s", output_dir, exc)
        sys.exit(1)

    logger.info("Output directory: %s", output_dir)
    logger.info("Processing %d dataset(s).\n", len(dataset_specs))

    # --- Process each dataset ---
    grand_total_images = 0
    grand_total_labels = 0

    for dataset_path, prefix in dataset_specs:
        # Law 4 — Fail Fast: dataset directory must exist
        if not dataset_path.is_dir():
            logger.error("Dataset directory not found: %s", dataset_path)
            sys.exit(1)

        logger.info("─" * 60)
        logger.info("Dataset: %s  (prefix: %s)", dataset_path, prefix)
        logger.info("─" * 60)

        summary = process_dataset(dataset_path, prefix, output_dir)

        # Per-split breakdown
        for split in SPLITS:
            s = summary["splits"].get(split)
            if s is None:
                continue
            total_split_skipped = (
                s["skipped_no_image"]
                + s["skipped_empty"]
                + s["skipped_malformed"]
                + s["skipped_duplicate"]
            )
            logger.info(
                "  %-8s images=%d  labels=%d  skipped=%d "
                "(no_img=%d, empty=%d, malformed=%d, dup=%d)",
                split + ":",
                s["images_copied"],
                s["labels_written"],
                total_split_skipped,
                s["skipped_no_image"],
                s["skipped_empty"],
                s["skipped_malformed"],
                s["skipped_duplicate"],
            )

        total_skipped = (
            summary["skipped_no_image"]
            + summary["skipped_empty"]
            + summary["skipped_malformed"]
            + summary["skipped_duplicate"]
        )
        logger.info(
            "  %-8s images=%d  labels=%d  skipped=%d "
            "(no_img=%d, empty=%d, malformed=%d, dup=%d)\n",
            "TOTAL:",
            summary["images_copied"],
            summary["labels_written"],
            total_skipped,
            summary["skipped_no_image"],
            summary["skipped_empty"],
            summary["skipped_malformed"],
            summary["skipped_duplicate"],
        )

        grand_total_images += summary["images_copied"]
        grand_total_labels += summary["labels_written"]

    # --- Grand total ---
    print(f"{'=' * 60}")
    print(f"  Unification Complete")
    print(f"{'=' * 60}")
    print(f"  Total images:  {grand_total_images}")
    print(f"  Total labels:  {grand_total_labels}")
    print(f"  Output:        {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
