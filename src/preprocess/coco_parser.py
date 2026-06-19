"""
COCO JSON annotation parser for Stage 1 YOLO format.

Reads an AnimeHeadsv3-style COCO dataset (one ``_annotations.coco.json`` per
``{train,valid,test}/`` split) and converts it into the flat YOLO format used
by the Stage 1 training pipeline.  Both COCO category IDs (0 and 1, both
labelled ``"head"``) are remapped to YOLO class 1 (head).

The library is **side-effect-light at the boundary**: it does not call
``sys.exit`` on invalid input — instead it raises ``CocoValidationError`` or
lets ``json.JSONDecodeError`` propagate, so the CLI wrapper decides how to
present the failure.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPLITS: tuple[str, ...] = ("train", "valid", "test")
COCO_ANNOTATION_FILENAME: str = "_annotations.coco.json"
IMAGE_PREFIX: str = "ahv3_"
HEAD_CLASS_ID: int = 1  # remapped YOLO class for both COCO category ids


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CocoValidationError(Exception):
    """Raised when the COCO input directory does not have the expected layout.

    The CLI wrapper catches this, logs a friendly message, and exits with
    a non-zero status.
    """


# ---------------------------------------------------------------------------
# Input validation (Law 4 — Fail Fast)
# ---------------------------------------------------------------------------

def validate_input_dir(input_dir: Path) -> dict[str, Path]:
    """Check that *input_dir* contains the expected ``{train,valid,test}/``
    directories, each with a ``_annotations.coco.json``.

    Returns ``{split_name: split_dir}`` on success.

    Raises:
        CocoValidationError: if required directories or annotation files
            are missing.
    """
    if not input_dir.is_dir():
        raise CocoValidationError(f"Input directory does not exist: {input_dir}")

    splits: dict[str, Path] = {}
    missing_splits: list[str] = []
    missing_anns: list[str] = []

    for split in SPLITS:
        split_dir = input_dir / split
        if not split_dir.is_dir():
            missing_splits.append(split)
            continue

        ann_path = split_dir / COCO_ANNOTATION_FILENAME
        if not ann_path.is_file():
            missing_anns.append(f"{split}/{COCO_ANNOTATION_FILENAME}")

        splits[split] = split_dir

    if missing_splits:
        raise CocoValidationError(
            f"Missing split directories in {input_dir}: "
            f"{', '.join(missing_splits)}"
        )

    if missing_anns:
        raise CocoValidationError(
            f"Missing annotation files: {', '.join(missing_anns)}"
        )

    return splits


# ---------------------------------------------------------------------------
# COCO parsing (Law 2 — Parse, Don't Validate — data parsed at boundary)
# ---------------------------------------------------------------------------

def load_image_map(images: list[dict]) -> dict[int, tuple[str, int, int]]:
    """Build ``image_id → (file_name, width, height)`` from a COCO ``images`` array.

    Silently skips malformed entries (bad image sizes, missing keys) so one
    corrupt image record does not abort the entire split.
    """
    mapping: dict[int, tuple[str, int, int]] = {}
    for img in images:
        img_id = img.get("id")
        file_name = img.get("file_name")
        width = img.get("width")
        height = img.get("height")

        if img_id is None or not file_name:
            logger.debug("Skipping image entry with missing id or file_name: %s", img)
            continue

        # Guard: non-positive dimensions are invalid (Law 1 — Early Exit)
        if not isinstance(width, (int, float)) or width <= 0:
            logger.debug("Skipping image %s: invalid width %s", img_id, width)
            continue
        if not isinstance(height, (int, float)) or height <= 0:
            logger.debug("Skipping image %s: invalid height %s", img_id, height)
            continue

        mapping[img_id] = (file_name, int(width), int(height))

    return mapping


# ---------------------------------------------------------------------------
# Bounding-box conversion
# ---------------------------------------------------------------------------

def coco_bbox_to_yolo(
    bbox: list[float],
    img_width: int,
    img_height: int,
    ann_id: int | None = None,
) -> tuple[tuple[float, float, float, float], bool] | None:
    """Convert a COCO ``[x, y, w, h]`` bbox to YOLO normalised
    ``(x_center, y_center, w, h)``.

    Returns ``None`` when the bbox is degenerate or lies entirely outside
    the image bounds (Fail Fast on garbage data).

    The second element of the return tuple indicates whether any coordinate
    was clamped to [0, 1] due to partial overflow beyond the image boundary.
    """
    if len(bbox) != 4:
        return None

    x, y, w, h = bbox

    # Guard: degenerate or out-of-bounds box
    if w <= 0 or h <= 0:
        return None
    if x + w <= 0 or y + h <= 0 or x >= img_width or y >= img_height:
        return None

    x_center = (x + w / 2.0) / img_width
    y_center = (y + h / 2.0) / img_height
    w_norm = w / img_width
    h_norm = h / img_height

    # Check for out-of-range values before clamping so partial-overflow
    # adjustments are visible for debugging.
    was_clamped = False
    if x_center < 0.0 or x_center > 1.0:
        logger.debug("Clamping x_center=%.4f for ann_id=%s", x_center, ann_id)
        was_clamped = True
    if y_center < 0.0 or y_center > 1.0:
        logger.debug("Clamping y_center=%.4f for ann_id=%s", y_center, ann_id)
        was_clamped = True
    if w_norm > 1.0:
        logger.debug("Clamping w_norm=%.4f for ann_id=%s", w_norm, ann_id)
        was_clamped = True
    if h_norm > 1.0:
        logger.debug("Clamping h_norm=%.4f for ann_id=%s", h_norm, ann_id)
        was_clamped = True

    # Clamp to [0, 1] for boxes that partially overflow
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w_norm = max(0.0, min(1.0, w_norm))
    h_norm = max(0.0, min(1.0, h_norm))

    return (x_center, y_center, w_norm, h_norm), was_clamped


def format_yolo_line(class_id: int, bbox: tuple[float, float, float, float]) -> str:
    """Serialize a single YOLO annotation line to text."""
    return f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"


# ---------------------------------------------------------------------------
# Single-split conversion (Law 3 — Atomic Predictability)
# ---------------------------------------------------------------------------

def convert_split(
    split_name: str,
    split_dir: Path,
    images_dst: Path,
    labels_dst: Path,
) -> dict[str, int]:
    """Convert one split (train/valid/test) from COCO-format to flat YOLO.

    Images are copied with an ``ahv3_`` prefix; labels are written with the
    same stem.  Images with no annotations are skipped.

    Returns a summary dict with keys: ``images_copied``, ``labels_written``,
    ``images_skipped``, ``bboxes_dropped``, ``bboxes_clamped``.

    Raises:
        json.JSONDecodeError: if the COCO JSON is malformed.
        TypeError: if required top-level keys are not lists.
    """
    ann_path = split_dir / COCO_ANNOTATION_FILENAME
    raw = ann_path.read_text(encoding="utf-8")
    coco = json.loads(raw)  # JSONDecodeError propagates to caller

    images_list = coco.get("images", [])
    annotations = coco.get("annotations", [])

    if not isinstance(images_list, list):
        raise TypeError(f"'images' key in {ann_path} is not a list")
    if not isinstance(annotations, list):
        raise TypeError(f"'annotations' key in {ann_path} is not a list")

    # Parse at boundary — build trusted image map (Law 2)
    image_map = load_image_map(images_list)

    if not image_map:
        logger.warning("  No valid image entries found in %s", ann_path)
        return {
            "images_copied": 0,
            "labels_written": 0,
            "images_skipped": 0,
            "bboxes_dropped": 0,
            "bboxes_clamped": 0,
        }

    # Group annotations by image_id (annotations may interleave in COCO JSON)
    ann_by_image: dict[int, list[dict]] = {}
    for ann in annotations:
        img_id = ann.get("image_id")
        if img_id is None:
            continue
        ann_by_image.setdefault(img_id, []).append(ann)

    images_copied = 0
    labels_written = 0
    images_skipped = 0
    bboxes_dropped = 0
    bboxes_clamped = 0

    for img_id, img_info in image_map.items():
        file_name, img_w, img_h = img_info
        anns = ann_by_image.get(img_id)

        # Guard: skip images without annotations (Law 1 — Early Exit)
        if not anns:
            images_skipped += 1
            continue

        # ---- Copy image ----
        src_path = split_dir / file_name
        if not src_path.is_file():
            logger.warning("  Image file missing, skipping: %s", src_path)
            images_skipped += 1
            continue

        dst_stem = f"{IMAGE_PREFIX}{Path(file_name).stem}"
        dst_img = images_dst / f"{dst_stem}.jpg"
        shutil.copy2(src_path, dst_img)
        images_copied += 1

        # ---- Convert and write labels ----
        label_lines: list[str] = []
        for ann in anns:
            bbox = ann.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                bboxes_dropped += 1
                continue

            # COCO bbox values can be int or float JSON numbers
            bbox_float = [float(v) for v in bbox]

            yolo_result = coco_bbox_to_yolo(bbox_float, img_w, img_h, ann.get("id"))
            if yolo_result is None:
                bboxes_dropped += 1
                continue

            yolo_bbox, was_clamped = yolo_result
            if was_clamped:
                bboxes_clamped += 1

            # Both category IDs 0 and 1 map to head (class 1)
            label_lines.append(format_yolo_line(HEAD_CLASS_ID, yolo_bbox))

        if not label_lines:
            # All bboxes for this image were degenerate → skip image entirely
            dst_img.unlink()
            images_copied -= 1
            images_skipped += 1
            continue

        dst_label = labels_dst / f"{dst_stem}.txt"
        dst_label.write_text("".join(label_lines), encoding="utf-8")
        labels_written += len(label_lines)

    return {
        "images_copied": images_copied,
        "labels_written": labels_written,
        "images_skipped": images_skipped,
        "bboxes_dropped": bboxes_dropped,
        "bboxes_clamped": bboxes_clamped,
    }
