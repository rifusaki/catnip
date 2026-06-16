"""
YOLOv8-format label remapper for Stage 1 (head class 0 → 1).

Reads datasets in the standard YOLOv8 layout (one ``train/``, ``valid/``,
``test/`` triple per variant with ``images/`` and ``labels/`` under each)
and writes a flat directory containing remapped labels and copied images
with source-specific filename prefixes.

Used for the anime head detection datasets (v1, v2, ani_face_detection)
that natively label heads as class 0 — these are remapped to Stage 1's
class 1 (head).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT: str = "catnip-data/data/staging/ah_yolo"
DEFAULT_DATA_DIR: str = "catnip-data/data/anime_head_detection"

# Known variant → source prefix used as filename prefix in flat output
DEFAULT_VARIANTS: list[tuple[str, str]] = [
    ("v1", "ahv1"),
    ("v2", "ahv2"),
    ("ani_face_detection", "ahaf"),
]

SPLITS: tuple[str, ...] = ("train", "valid", "test")

# Ordered by preference — YOLO datasets are almost always .jpg
IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


# ---------------------------------------------------------------------------
# Guards (Law 4 — Fail Fast, Fail Loud)
# ---------------------------------------------------------------------------

def ensure_writable(directory: Path) -> None:
    """Create *directory* (and parents).

    Raises:
        PermissionError: if the directory cannot be created due to permissions.
        OSError: for other filesystem errors.
    """
    directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Parse & Validate (Law 2 — Parse at the boundary)
# ---------------------------------------------------------------------------

def parse_label_line(
    line: str,
    line_num: int,
    source: str,
) -> tuple[int, float, float, float, float] | None:
    """Parse a YOLO label line into ``(class_id, x_center, y_center, width, height)``.

    Returns ``None`` for blank or malformed lines and logs a warning.
    """
    stripped = line.strip()
    if not stripped:
        return None

    parts = stripped.split()
    if len(parts) != 5:
        logger.warning(
            "  [%s:%d] Malformed label — expected 5 fields, got %d: %r",
            source, line_num, len(parts), stripped,
        )
        return None

    try:
        class_id = int(parts[0])
        coords = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
    except ValueError:
        logger.warning(
            "  [%s:%d] Malformed label — non-numeric fields: %r",
            source, line_num, stripped,
        )
        return None

    return (class_id, *coords)


def remap_class(class_id: int, source: str) -> int:
    """Remap existing class IDs to unified Stage 1 format.

    Source ``0`` (head) maps to ``1``.  Unexpected values return the
    sentinel ``-1`` to signal that the entire label file should be dropped.
    """
    if class_id == 0:
        return 1
    logger.error("  [%s] Unexpected class ID %d — returning sentinel.", source, class_id)
    return -1


# ---------------------------------------------------------------------------
# File resolution helpers
# ---------------------------------------------------------------------------

def find_image(images_dir: Path, stem: str) -> Path | None:
    """Return the image file matching *stem* in *images_dir*, or ``None``."""
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def resolve_variant(rel_path: str, project_root: Path) -> tuple[Path, str]:
    """Resolve a relative dataset path to an absolute ``Path`` and its source prefix.

    For known variants (v1, v2, ani_face_detection), returns the fixed prefix.
    For unknown paths, infers the prefix from the directory name.
    """
    dataset_path = (project_root / rel_path).resolve()

    variant_name = dataset_path.name
    for known_variant, known_prefix in DEFAULT_VARIANTS:
        if variant_name == known_variant:
            return dataset_path, known_prefix

    return dataset_path, variant_name


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_split(
    dataset_path: Path,
    split: str,
    prefix: str,
    images_dir: Path,
    labels_dir: Path,
) -> dict:
    """Process a single split (train/valid/test) of one dataset.

    Returns a per-split summary dict with keys ``images_copied``,
    ``labels_written``, ``skipped_no_image``, ``skipped_empty``,
    ``skipped_malformed``, and ``skipped_duplicate``.
    """
    split_labels = dataset_path / split / "labels"
    split_images = dataset_path / split / "images"

    summary = {
        "images_copied": 0,
        "labels_written": 0,
        "skipped_no_image": 0,
        "skipped_empty": 0,
        "skipped_malformed": 0,
        "skipped_duplicate": 0,
    }

    # Law 1 — Early Exit: split directory must exist
    if not split_labels.is_dir():
        logger.warning("  Split '%s': labels directory not found — %s", split, split_labels)
        return summary

    for label_file in sorted(split_labels.glob("*.txt")):
        stem = label_file.stem
        source = f"{prefix}/{split}/{stem}"

        # Law 1 — Early Exit: no image to pair with this label
        image_file = find_image(split_images, stem)
        if image_file is None:
            summary["skipped_no_image"] += 1
            continue

        # Parse and remap every line
        remapped_lines: list[str] = []
        lines_attempted = 0
        had_unknown_class = False
        with open(label_file, encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, 1):
                lines_attempted += 1
                parsed = parse_label_line(line, line_num, source)
                if parsed is None:
                    continue
                class_id, xc, yc, w, h = parsed
                new_class = remap_class(class_id, source)
                if new_class == -1:
                    had_unknown_class = True
                    continue  # drop this annotation line
                remapped_lines.append(
                    f"{new_class} {xc:.6g} {yc:.6g} {w:.6g} {h:.6g}\n"
                )

        # Law 4 — Fail Loud: unknown class ID poisons the entire label file
        if had_unknown_class:
            summary["skipped_malformed"] += 1
            continue

        # Law 1 — Early Exit: label file had no usable content
        if not remapped_lines:
            if lines_attempted == 0:
                summary["skipped_empty"] += 1
            else:
                summary["skipped_malformed"] += 1
            continue

        # Law 1 — Early Exit: skip duplicate (same stem already in output)
        new_stem = f"{prefix}_{stem}"
        image_dest = images_dir / f"{new_stem}{image_file.suffix}"
        if image_dest.exists():
            logger.warning(
                "  [%s] Duplicate stem '%s' — already exists in output; skipping.",
                source, new_stem,
            )
            summary["skipped_duplicate"] += 1
            continue

        # Write remapped label
        label_dest = labels_dir / f"{new_stem}.txt"
        with open(label_dest, "w", encoding="utf-8") as fh:
            fh.writelines(remapped_lines)

        # Copy image (shutil.copy2 preserves metadata)
        shutil.copy2(image_file, image_dest)

        summary["images_copied"] += 1
        summary["labels_written"] += 1

    return summary


def process_dataset(
    dataset_path: Path,
    prefix: str,
    output_dir: Path,
) -> dict:
    """Convert one YOLO dataset variant into the unified flat format.

    Returns a summary dictionary::

        {
            "dataset":          str,    # variant name (e.g. "v1")
            "prefix":           str,    # filename prefix (e.g. "ahv1")
            "images_copied":    int,
            "labels_written":   int,
            "skipped_no_image": int,
            "skipped_empty":    int,
            "skipped_malformed":int,
            "splits":           {split: {...}, ...},
        }
    """
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    ensure_writable(images_dir)
    ensure_writable(labels_dir)

    total_images = 0
    total_labels = 0
    total_skipped_no_image = 0
    total_skipped_empty = 0
    total_skipped_malformed = 0
    total_skipped_duplicate = 0
    splits_summary: dict[str, dict] = {}

    for split in SPLITS:
        split_summary = process_split(dataset_path, split, prefix, images_dir, labels_dir)
        splits_summary[split] = split_summary
        total_images += split_summary["images_copied"]
        total_labels += split_summary["labels_written"]
        total_skipped_no_image += split_summary["skipped_no_image"]
        total_skipped_empty += split_summary["skipped_empty"]
        total_skipped_malformed += split_summary["skipped_malformed"]
        total_skipped_duplicate += split_summary["skipped_duplicate"]

    return {
        "dataset": dataset_path.name,
        "prefix": prefix,
        "images_copied": total_images,
        "labels_written": total_labels,
        "skipped_no_image": total_skipped_no_image,
        "skipped_empty": total_skipped_empty,
        "skipped_malformed": total_skipped_malformed,
        "skipped_duplicate": total_skipped_duplicate,
        "splits": splits_summary,
    }
