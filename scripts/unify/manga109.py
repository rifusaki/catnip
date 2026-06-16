#!/usr/bin/env python3
"""
Thin CLI wrapper around :mod:`src.preprocess.manga109_parser`.

Converts Manga109 XML annotations to YOLO format for catnip Stage 1 training.

Manga109 provides bounding-box annotations (body, face) in per-title XML files
with pixel coordinates.  This script normalises coordinates to YOLO format and
remaps annotation types to the Stage 1 class schema: body=0, head=1, face=2.
(Manga109 only has body and face; no head data is produced.)

The library does the XML parsing and YOLO conversion; this wrapper handles
CLI argument parsing, input validation, output directory creation, and
summary printing.

Usage::

    python scripts/unify/manga109.py

    python scripts/unify/manga109.py \\
        --annotations-dir /Volumes/rifuSSD/catnip-data/data/manga109/annotations \\
        --images-dir /Volumes/rifuSSD/catnip-data/data/manga109/images \\
        --output-dir /Volumes/rifuSSD/catnip-data/data/staging/manga109

    python scripts/unify/manga109.py --verbose
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running the script directly from scripts/unify/ without an installed package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import settings
from src.preprocess.manga109_parser import parse_xml_annotations, process_title

logger = logging.getLogger("unify_manga109")


# ---------------------------------------------------------------------------
# Summary printer (Law 5 — Intentional Naming)
# ---------------------------------------------------------------------------

def _print_summary(overall: dict, xml_file_count: int) -> None:
    """Print a human-readable summary of the conversion."""
    header = " Manga109 → YOLO Conversion Summary "
    print(f"\n{'=' * 60}")
    print(f"{header:=^60}")
    print(f"{'=' * 60}")

    print(f"\n  XML files parsed:       {xml_file_count:>6}")
    print(f"  Pages with annotations: {overall['pages_processed']:>6}")
    print(f"  Images copied:          {overall['images_copied']:>6}")
    print(f"\n  Annotations:")
    print(f"    body (class 0):       {overall['body_count']:>6}")
    print(f"    face (class 2):       {overall['face_count']:>6}")
    print(f"    head (class 1):       {overall.get('head_count', 0):>6}  "
          f"(always 0 — Manga109 has no head class)")
    print(f"    skipped (OOB/deg):    {overall.get('skipped_oob', 0):>6}  "
          f"(out-of-bounds or degenerate boxes)")

    # Count XML files with no usable annotations (Law 1 — surface early)
    skipped_titles = overall.get("skipped_titles", 0)
    if skipped_titles:
        print(f"\n  Titles skipped:         {skipped_titles:>6}  "
              f"(no images directory or no annotations)")

    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Manga109 XML annotations to YOLO format.",
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=settings.paths.manga109_dir / "annotations",
        help="Directory containing Manga109 XML annotation files "
        "(default from config: paths.manga109_dir/annotations).",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=settings.paths.manga109_dir / "images",
        help="Directory containing per-title image subdirectories "
        "(default from config: paths.manga109_dir/images).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.paths.data / "data" / "staging" / "manga109",
        help="Output directory for converted YOLO dataset "
        "(default from config: paths.data/data/staging/manga109).",
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

    logger.info("Annotations: %s", args.annotations_dir)
    logger.info("Images:      %s", args.images_dir)
    logger.info("Output:      %s", args.output_dir)

    # --- Validate inputs (Law 4 — Fail Fast) ---
    if not args.annotations_dir.is_dir():
        logger.error("Annotations directory not found: %s", args.annotations_dir)
        sys.exit(1)
    if not args.images_dir.is_dir():
        logger.error("Images directory not found: %s", args.images_dir)
        sys.exit(1)

    # --- Prepare output directories ---
    images_out = args.output_dir / "images"
    labels_out = args.output_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    # --- Discover XML files ---
    xml_files = sorted(args.annotations_dir.glob("*.xml"))
    if not xml_files:
        logger.error("No XML files found in %s", args.annotations_dir)
        sys.exit(1)

    logger.info("Found %d XML annotation files", len(xml_files))

    # --- Process each XML ---
    overall: dict = {
        "pages_processed": 0,
        "images_copied": 0,
        "body_count": 0,
        "face_count": 0,
        "head_count": 0,
        "skipped_oob": 0,
        "skipped_titles": 0,
    }

    for xml_path in xml_files:
        # Parse XML at the boundary (Law 2)
        title, pages, skipped_oob = parse_xml_annotations(xml_path)

        # Guard: no pages with annotations for this title
        if not pages:
            logger.debug("  No annotations found in '%s' — skipping", title)
            continue

        logger.debug("Parsing '%s': %d pages with annotations", title, len(pages))

        details = process_title(
            title, pages, args.images_dir, images_out, labels_out, skipped_oob,
        )

        # Guard: entire title had no valid image/annotation pairs
        if details["images_copied"] == 0:
            overall["skipped_titles"] += 1
            logger.info("  %s: no images processed (check paths)", title)
            continue

        overall["pages_processed"] += details["pages_processed"]
        overall["images_copied"] += details["images_copied"]
        overall["body_count"] += details["body_count"]
        overall["face_count"] += details["face_count"]
        overall["head_count"] += details["head_count"]
        overall["skipped_oob"] += details["skipped_oob"]

        logger.info(
            "  %s: %d pages, %d body, %d face",
            title,
            details["pages_processed"],
            details["body_count"],
            details["face_count"],
        )

    # --- Summary ---
    _print_summary(overall, len(xml_files))


if __name__ == "__main__":
    main()
