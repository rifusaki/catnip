#!/usr/bin/env python3
"""
Convert izutsumi Label Studio per-annotation JSON exports to YOLO format.

Reads the 76 per-annotation JSON files (chapters 31-106), converts bounding-box
coordinates from Label Studio percentage format (0-100, top-left) to YOLO
normalised format (0-1, centre-relative), remaps labels to the 3-class Stage-1
scheme (body=0, head=1, face=2), and copies the source manga images with
renamed filenames into a flat output directory.

The heavy lifting (annotation parsing, coordinate conversion, label
writing) is delegated to :func:`src.preprocess.convert_labels.convert_annotations_directory`.
This wrapper only handles CLI/IO orchestration.

Usage::

    python scripts/unify/izutsumi.py

    python scripts/unify/izutsumi.py --input-dir /Volumes/rifuSSD/catnip-data/data/izutsumi/annotations \\
        --manga-dir /Volumes/rifuSSD/catnip-data/data/izutsumi/manga \\
        --output-dir /Volumes/rifuSSD/catnip-data/data/staging/izutsumi

    python scripts/unify/izutsumi.py --verbose
"""

import argparse
import logging
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

# Allow running the script directly from scripts/unify/ without an installed package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import settings
from src.preprocess.convert_labels import convert_annotations_directory

logger = logging.getLogger("unify_izutsumi")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAGE1_CLASSES: dict[str, int] = {"body": 0, "head": 1, "face": 2}
OUTPUT_IMAGE_PREFIX = "izutsumi"


# ---------------------------------------------------------------------------
# Coordinate extraction helpers (Law 2 — Parse, Don't Validate)
# ---------------------------------------------------------------------------

def _extract_volume_page(rel_path: str) -> tuple[str, str] | None:
    """Extract volume and page number from a relative manga path.

    Args:
        rel_path: e.g. ``"v09/0043.jpg"`` or ``"v14/0087.png"``.

    Returns:
        ``(volume_dir, stem)`` on success — e.g. ``("v09", "0043")``.
        ``None`` if the path doesn't match the expected ``VOLUME/PAGE.EXT``
        pattern.
    """
    parts = rel_path.strip().split("/")

    # Guard: expected exactly two components (Law 1 — Early Exit)
    if len(parts) != 2:
        return None

    volume_dir, filename = parts

    # Guard: volume dir must look like "vNN" (Law 4 — Fail Fast on bad data)
    if not volume_dir.startswith("v") or not volume_dir[1:].isdigit():
        return None

    stem = Path(filename).stem

    # Guard: stem must be non-empty and numeric-ish
    if not stem or not stem.isdigit():
        return None

    return volume_dir, stem


def _rel_path_to_new_name(rel_path: str) -> str | None:
    """Convert a manga relative path to the flat output filename stem.

    ``"v09/0043.jpg"`` → ``"izutsumi_v09_0043"``

    Returns ``None`` if the path cannot be parsed.
    """
    parsed = _extract_volume_page(rel_path)
    if parsed is None:
        return None
    volume_dir, stem = parsed
    return f"{OUTPUT_IMAGE_PREFIX}_{volume_dir}_{stem}"


# ---------------------------------------------------------------------------
# File-system operations (Law 3 — Atomic Predictability)
# ---------------------------------------------------------------------------

def _collect_label_files(temp_dir: Path) -> dict[str, Path]:
    """Walk a temp directory containing YOLO labels and build a mapping.

    Labelled images are identified by finding ``.txt`` label files written
    by ``convert_annotations_directory``.  The keys are the original relative
    manga paths (e.g. ``"v09/0043.jpg"``) and the values are the absolute
    paths to the corresponding label ``.txt`` files in *temp_dir*.

    Returns an empty dict if no label files were generated.
    """
    mapping: dict[str, Path] = {}

    for txt_path in sorted(temp_dir.rglob("*.txt")):
        # Reconstruct the relative source path from the label path inside the
        # temp directory.  The conversion function writes labels at the same
        # relative location as the source image, so:
        #   temp_dir / "v09/0043.txt"  →  "v09/0043.jpg"
        rel_to_temp = txt_path.relative_to(temp_dir)
        rel_image = rel_to_temp.with_suffix(".jpg").as_posix()
        mapping[rel_image] = txt_path

    return mapping


def _count_class_distribution(labels_dir: Path) -> Counter:
    """Tally class IDs across every YOLO label file in *labels_dir*.

    Each line in a label file begins with ``class_id`` (0, 1, or 2).
    """
    counter: Counter = Counter()

    for label_file in sorted(labels_dir.glob("*.txt")):
        try:
            raw = label_file.read_text(encoding="utf-8")
        except OSError:
            continue

        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            fields = stripped.split()
            if not fields:
                continue
            try:
                class_id = int(fields[0])
            except ValueError:
                continue
            counter[class_id] += 1

    return counter


# ---------------------------------------------------------------------------
# Core workflow
# ---------------------------------------------------------------------------

def _copy_images_and_rename_labels(
    manga_dir: Path,
    temp_labels_dir: Path,
    output_dir: Path,
) -> dict:
    """Copy manga images and rename YOLO labels into the flat output structure.

    Args:
        manga_dir: Root of the izutsumi manga directory (contains ``v01/``
            through ``v14/``).
        temp_labels_dir: Temporary directory containing YOLO labels written
            by ``convert_annotations_directory`` (preserves source sub-path).
        output_dir: Target directory.  Creates ``images/`` and ``labels/``
            subdirectories inside it.

    Returns:
        A dict with keys ``images_copied``, ``labels_written``,
        ``images_missing``, and ``skipped_no_annotations``.
    """
    images_dst = output_dir / "images"
    labels_dst = output_dir / "labels"
    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    label_mapping = _collect_label_files(temp_labels_dir)

    # Guard: no labels generated at all (Law 1 — Early Exit)
    if not label_mapping:
        logger.warning("No label files were generated — nothing to copy.")
        return {
            "images_copied": 0,
            "labels_written": 0,
            "images_missing": 0,
            "skipped_no_annotations": 0,
        }

    images_copied = 0
    labels_written = 0
    images_missing = 0
    skipped_no_annotations = 0

    for rel_image, label_src in sorted(label_mapping.items()):
        new_stem = _rel_path_to_new_name(rel_image)
        if new_stem is None:
            logger.warning("Skipping %r: could not extract volume/page.", rel_image)
            skipped_no_annotations += 1
            continue

        # --- Check source image ---
        image_src = manga_dir / rel_image
        if not image_src.is_file():
            logger.warning("Image not found, skipping: %s", image_src)
            images_missing += 1
            continue

        # --- Copy image with renamed stem ---
        new_image_name = f"{new_stem}{image_src.suffix}"
        image_dst = images_dst / new_image_name
        shutil.copy2(image_src, image_dst)
        images_copied += 1

        # --- Copy label with renamed stem ---
        label_dst = labels_dst / f"{new_stem}.txt"
        shutil.copy2(label_src, label_dst)
        labels_written += 1

    return {
        "images_copied": images_copied,
        "labels_written": labels_written,
        "images_missing": images_missing,
        "skipped_no_annotations": skipped_no_annotations,
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(
    conversion_result: dict,
    copy_result: dict,
    output_dir: Path,
) -> None:
    """Print a human-readable summary of the conversion and image copying."""
    header = " izutsumi → YOLO Conversion Summary "
    print(f"\n{'=' * 64}")
    print(f"{header:=^64}")
    print(f"{'=' * 64}")

    annotation_files = conversion_result.get("file_count", 0)
    annotation_skipped = conversion_result.get("skipped_json", 0)
    label_count = conversion_result.get("label_count", 0)
    images_copied = copy_result.get("images_copied", 0)
    labels_written = copy_result.get("labels_written", 0)
    images_missing = copy_result.get("images_missing", 0)
    skipped_no_annotations = copy_result.get("skipped_no_annotations", 0)

    print(f"\n  Annotation files processed:  {annotation_files:>12}")
    print(f"  Annotation files skipped:    {annotation_skipped:>12}")
    print(f"  Unique source images found:   {label_count:>12}")
    print(f"  Images copied:               {images_copied:>12}")
    print(f"  Images missing (skipped):    {images_missing:>12}")
    print(f"  Skipped (no annotations):    {skipped_no_annotations:>12}")

    # Class distribution
    labels_dir = output_dir / "labels"
    if labels_dir.is_dir():
        distribution = _count_class_distribution(labels_dir)
        print(f"\n  Class distribution (bbox count):")
        for class_id in sorted(STAGE1_CLASSES.values()):
            name = {v: k for k, v in STAGE1_CLASSES.items()}.get(class_id, f"?{class_id}")
            count = distribution.get(class_id, 0)
            print(f"    {class_id} ({name:6s}):  {count:>8}")
        print(f"    {'total':12s}:  {sum(distribution.values()):>8}")

    print(f"\n  Output written to: {output_dir}")
    print(f"{'=' * 64}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert izutsumi Label Studio annotations to YOLO format.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=settings.paths.izutsumi_annotations_dir,
        help="Directory containing per-annotation JSON files "
        "(default from config: paths.data/data/izutsumi/annotations).",
    )
    parser.add_argument(
        "--manga-dir",
        type=Path,
        default=settings.paths.izutsumi_manga_dir,
        help="Root of the izutsumi manga image tree (default from config: "
        "paths.data/data/izutsumi/manga).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.paths.data / "data" / "staging" / "izutsumi",
        help="Output directory for the YOLO dataset "
        "(default from config: paths.data/data/staging/izutsumi).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )

    args = parser.parse_args()

    # --- Setup logging ---
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)-7s %(name)-32s %(message)s",
    )

    logger.info("Input annotations:  %s", args.input_dir)
    logger.info("Manga images:       %s", args.manga_dir)
    logger.info("Output directory:   %s", args.output_dir)

    # --- Validate inputs (Law 4 — Fail Fast) ---
    if not args.input_dir.is_dir():
        logger.error("Annotations directory not found: %s", args.input_dir)
        sys.exit(1)
    if not args.manga_dir.is_dir():
        logger.error("Manga directory not found: %s", args.manga_dir)
        sys.exit(1)

    # --- Step 1: Convert annotations → YOLO labels in a temp directory ---
    # The library converter groups by source image and writes labels
    # preserving the source sub-path (e.g. v09/0043.txt).
    logger.info("Converting annotations to YOLO format...")

    with tempfile.TemporaryDirectory(prefix="izutsumi_labels_") as temp_dir:
        temp_path = Path(temp_dir)

        conversion_result = convert_annotations_directory(
            dir_path=str(args.input_dir),
            output_dir=str(temp_path),
            class_map=STAGE1_CLASSES,
        )

        logger.info(
            "Conversion complete: %d annotation files → %d unique label files.",
            conversion_result.get("file_count", 0),
            conversion_result.get("label_count", 0),
        )

        # --- Step 2: Copy images & rename labels into flat output structure ---
        logger.info("Copying images and renaming labels to flat structure...")

        copy_result = _copy_images_and_rename_labels(
            manga_dir=args.manga_dir,
            temp_labels_dir=temp_path,
            output_dir=args.output_dir,
        )

    # --- Step 3: Print summary ---
    _print_summary(conversion_result, copy_result, args.output_dir)


if __name__ == "__main__":
    main()
