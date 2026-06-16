"""
Pre-slice YOLO training data using the same parameters as SAHI inference.

Motivation
----------
SAHI slices a full manga page into overlapping 640x640 patches at inference
time. If YOLO is trained on the unsliced full pages, it never sees the
"zoomed-in" crops the inference path produces — a train/inference scale
mismatch that hurts small-head mAP.

This module slices every training image with the SAME parameters SAHI uses
at inference and remaps each YOLO label into the slice it lands in. The
output is a new YOLO dataset whose images and labels are byte-identical
(per-box) to what the model will see during SAHI sliced prediction.

Label math
----------
YOLO labels are normalised ``(class_id, x_center, y_center, width, height)``
in [0, 1] over the source image. For each slice ``S = (sx1, sy1, sx2, sy2)``
in source pixel coords, a label box ``B`` is remapped as follows:

1. Convert ``B`` to absolute pixel coords on the source image.
2. Compute the rectangle ``I = B ∩ S`` (in source pixel coords).
3. If ``area(I) / area(B) < min_area_ratio`` → drop the box.
4. Translate ``I`` to slice-relative pixel coords (subtract ``sx1``, ``sy1``).
5. Normalise by the slice's OUTPUT dimensions in source pixels
   (``sx2 - sx1``, ``sy2 - sy1``).

Step 5 deliberately normalises by the slice's source-pixel extent (NOT by
``slice_height x slice_width``), because that matches what ``sahi.slicing.
slice_image`` actually produces — a non-padded numpy crop. YOLO's
letterbox transform at training time remaps the labels to the letterboxed
frame, which is the same letterbox that runs at inference inside SAHI.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image
from tqdm import tqdm

from sahi.slicing import get_slice_bboxes

logger = logging.getLogger(__name__)


IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class YoloBox:
    """A single YOLO label, normalised over the source image (0..1)."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_line(self) -> str:
        """Format as a single YOLO ``.txt`` line."""
        return (
            f"{self.class_id} "
            f"{self.x_center:.6f} "
            f"{self.y_center:.6f} "
            f"{self.width:.6f} "
            f"{self.height:.6f}"
        )


@dataclass(frozen=True)
class SliceStats:
    """Aggregate stats for a single image's slicing."""

    slices_out: int = 0
    kept_boxes: int = 0
    dropped_boxes: int = 0

    def __add__(self, other: "SliceStats") -> "SliceStats":
        return SliceStats(
            slices_out=self.slices_out + other.slices_out,
            kept_boxes=self.kept_boxes + other.kept_boxes,
            dropped_boxes=self.dropped_boxes + other.dropped_boxes,
        )


# ---------------------------------------------------------------------------
# Pure helpers (testable, no I/O)
# ---------------------------------------------------------------------------


def remap_box_to_slice(
    box: YoloBox,
    image_width: int,
    image_height: int,
    slice_bbox: tuple[int, int, int, int],
    min_area_ratio: float,
) -> YoloBox | None:
    """Remap a YOLO box from source-image coords to slice-relative coords.

    Returns ``None`` when the box has no intersection with the slice or the
    intersection is smaller than ``min_area_ratio`` of the original box.

    Args:
        box: Source-image-normalised YOLO box.
        image_width: Source image width in pixels.
        image_height: Source image height in pixels.
        slice_bbox: ``(x1, y1, x2, y2)`` in source pixel coords.
        min_area_ratio: Boxes whose kept area is below this fraction of
            the original are dropped (boundary truncation).
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Invalid image size: {image_width}x{image_height}")
    if min_area_ratio < 0 or min_area_ratio > 1:
        raise ValueError(f"min_area_ratio must be in [0, 1], got {min_area_ratio}")

    sx1, sy1, sx2, sy2 = slice_bbox
    if sx2 <= sx1 or sy2 <= sy1:
        raise ValueError(f"Degenerate slice_bbox: {slice_bbox}")

    bw = box.width * image_width
    bh = box.height * image_height
    if bw <= 0 or bh <= 0:
        return None

    bx1 = box.x_center * image_width - bw / 2.0
    by1 = box.y_center * image_height - bh / 2.0
    bx2 = bx1 + bw
    by2 = by1 + bh

    ix1 = max(bx1, sx1)
    iy1 = max(by1, sy1)
    ix2 = min(bx2, sx2)
    iy2 = min(by2, sy2)

    if ix2 <= ix1 or iy2 <= iy1:
        return None

    orig_area = bw * bh
    kept_area = (ix2 - ix1) * (iy2 - iy1)
    if orig_area <= 0 or kept_area <= 0:
        return None
    if kept_area / orig_area < min_area_ratio:
        return None

    slice_w = sx2 - sx1
    slice_h = sy2 - sy1
    new_xc = (ix1 + ix2) / 2.0 - sx1
    new_yc = (iy1 + iy2) / 2.0 - sy1
    new_w = ix2 - ix1
    new_h = iy2 - iy1

    return YoloBox(
        class_id=box.class_id,
        x_center=new_xc / slice_w,
        y_center=new_yc / slice_h,
        width=new_w / slice_w,
        height=new_h / slice_h,
    )


def parse_yolo_label_file(path: Path) -> list[YoloBox]:
    """Read a YOLO ``.txt`` label file. Skips malformed lines silently."""
    boxes: list[YoloBox] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Cannot read label file %s: %s", path, exc)
        return boxes

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            logger.warning("Malformed label in %s: %r (expected 5 fields)", path, line)
            continue
        try:
            class_id = int(parts[0])
            xc, yc, w, h = (float(p) for p in parts[1:5])
        except ValueError:
            logger.warning("Non-numeric label in %s: %r", path, line)
            continue
        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
            logger.warning("Out-of-range center in %s: %r (must be in [0, 1])", path, line)
            continue
        if w <= 0 or h <= 0 or w > 1.0 or h > 1.0:
            logger.warning("Degenerate/out-of-range box in %s: %r", path, line)
            continue
        boxes.append(YoloBox(class_id, xc, yc, w, h))

    return boxes


def write_yolo_label_file(path: Path, boxes: Sequence[YoloBox]) -> None:
    """Write boxes to a YOLO ``.txt`` label file.

    Always writes the file, even when ``boxes`` is empty (empty file = a
    valid negative sample for YOLO).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(b.to_line() for b in boxes)
    if boxes:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def make_slice_filename(stem: str, ext: str, slice_bbox: tuple[int, int, int, int]) -> str:
    """Build a deterministic slice filename encoding the source-pixel bbox.

    Example: ``"v09_0043_0_512_640_1152.jpg"`` — slice spanning x=0..640,
    y=512..1152 in the source image.
    """
    return f"{stem}_{slice_bbox[0]}_{slice_bbox[1]}_{slice_bbox[2]}_{slice_bbox[3]}{ext}"


# ---------------------------------------------------------------------------
# Single-image worker (module-level so ProcessPoolExecutor can pickle it)
# ---------------------------------------------------------------------------


def _slice_single_image(
    image_path: str,
    label_path: str,
    out_image_dir: str,
    out_label_dir: str,
    slice_height: int,
    slice_width: int,
    overlap_height_ratio: float,
    overlap_width_ratio: float,
    min_area_ratio: float,
) -> SliceStats:
    """Slice one ``(image, label)`` pair and write outputs.

    Returns counts for the caller. The output filenames are deterministic
    (``{stem}_{x1}_{y1}_{x2}_{y2}{ext}``) so re-runs are idempotent.
    """
    image_path = Path(image_path)
    label_path = Path(label_path)
    out_image_dir = Path(out_image_dir)
    out_label_dir = Path(out_label_dir)

    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    boxes = parse_yolo_label_file(label_path) if label_path.is_file() else []

    try:
        with Image.open(image_path) as pil_img:
            pil_img.load()
            img_w, img_h = pil_img.size
            mode = pil_img.mode
            bboxes = get_slice_bboxes(
                image_height=img_h,
                image_width=img_w,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_height_ratio,
                overlap_width_ratio=overlap_width_ratio,
                auto_slice_resolution=False,
            )

            total_kept = 0
            total_dropped = 0

            for bbox in bboxes:
                sx1, sy1, sx2, sy2 = bbox
                slice_img = pil_img.crop((sx1, sy1, sx2, sy2))
                if slice_img.mode != mode:
                    slice_img = slice_img.convert(mode)

                slice_img.save(out_image_dir / make_slice_filename(
                    image_path.stem,
                    image_path.suffix.lower(),
                    bbox,
                ))

                kept: list[YoloBox] = []
                for b in boxes:
                    remapped = remap_box_to_slice(b, img_w, img_h, bbox, min_area_ratio)
                    if remapped is not None:
                        kept.append(remapped)
                    else:
                        total_dropped += 1
                total_kept += len(kept)

                write_yolo_label_file(
                    out_label_dir / make_slice_filename(image_path.stem, ".txt", bbox),
                    kept,
                )

        return SliceStats(
            slices_out=len(bboxes),
            kept_boxes=total_kept,
            dropped_boxes=total_dropped,
        )

    except Exception as exc:
        logger.error("Failed to slice %s: %s", image_path, exc)
        return SliceStats()


# ---------------------------------------------------------------------------
# Dataset driver
# ---------------------------------------------------------------------------


def discover_image_label_pairs(
    images_dir: Path,
    labels_dir: Path,
) -> list[tuple[Path, Path]]:
    """Find ``(image, label)`` pairs under *images_dir* and *labels_dir*.

    The two directories must be mirrored (same relative stems). Only images
    with a corresponding label file are returned.
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    pairs: list[tuple[Path, Path]] = []
    skipped = 0
    for img_path in sorted(images_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        lbl_path = labels_dir / (img_path.stem + ".txt")
        if not lbl_path.is_file():
            skipped += 1
            continue
        pairs.append((img_path, lbl_path))

    if skipped:
        logger.info(
            "Skipped %d image(s) without label file under %s",
            skipped, images_dir,
        )
    return pairs


def slice_split(
    images_dir: Path | str,
    labels_dir: Path | str,
    output_images_dir: Path | str,
    output_labels_dir: Path | str,
    slice_height: int = 640,
    slice_width: int = 640,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    min_area_ratio: float = 0.3,
    num_workers: int = 1,
) -> dict:
    """Slice one (train/val/test) split of a YOLO dataset.

    Args:
        images_dir: Source images directory.
        labels_dir: Source labels directory.
        output_images_dir: Where to write sliced images.
        output_labels_dir: Where to write sliced labels.
        slice_height, slice_width: SAHI slice size in pixels.
        overlap_height_ratio, overlap_width_ratio: SAHI overlap ratios.
        min_area_ratio: Boxes whose kept area is below this fraction are dropped.
        num_workers: Number of parallel processes. 1 = serial.

    Returns:
        Dict with per-split stats: ``images_in, slices_out, kept_boxes,
        dropped_boxes, elapsed_sec``.
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_images_dir = Path(output_images_dir)
    output_labels_dir = Path(output_labels_dir)

    pairs = discover_image_label_pairs(images_dir, labels_dir)
    if not pairs:
        logger.warning("No image/label pairs found in %s", images_dir)
        return {
            "images_in": 0,
            "slices_out": 0,
            "kept_boxes": 0,
            "dropped_boxes": 0,
            "elapsed_sec": 0.0,
        }

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Slicing %d images: %s → %s (slice=%dx%d, overlap=%.2f, min_area=%.2f, workers=%d)",
        len(pairs), images_dir, output_images_dir,
        slice_height, slice_width,
        overlap_height_ratio, min_area_ratio, num_workers,
    )

    t0 = time.monotonic()
    total = SliceStats()

    if num_workers <= 1:
        for img_p, lbl_p in tqdm(pairs, desc=f"slice:{images_dir.name}"):
            stats = _slice_single_image(
                str(img_p), str(lbl_p),
                str(output_images_dir), str(output_labels_dir),
                slice_height, slice_width,
                overlap_height_ratio, overlap_width_ratio,
                min_area_ratio,
            )
            total = total + stats
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {
                pool.submit(
                    _slice_single_image,
                    str(img_p), str(lbl_p),
                    str(output_images_dir), str(output_labels_dir),
                    slice_height, slice_width,
                    overlap_height_ratio, overlap_width_ratio,
                    min_area_ratio,
                ): (img_p, lbl_p)
                for img_p, lbl_p in pairs
            }
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f"slice:{images_dir.name}"):
                stats = fut.result()
                total = total + stats

    elapsed = time.monotonic() - t0

    stats = {
        "images_in": len(pairs),
        "slices_out": total.slices_out,
        "kept_boxes": total.kept_boxes,
        "dropped_boxes": total.dropped_boxes,
        "elapsed_sec": round(elapsed, 2),
    }
    logger.info(
        "Done: %d images → %d slices, kept=%d dropped=%d in %.1fs",
        stats["images_in"], stats["slices_out"],
        stats["kept_boxes"], stats["dropped_boxes"], stats["elapsed_sec"],
    )
    return stats


def slice_dataset(
    source_images_dir: Path | str,
    source_labels_dir: Path | str,
    output_dir: Path | str,
    splits: Sequence[str] = ("train", "val", "test"),
    slice_height: int = 640,
    slice_width: int = 640,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    min_area_ratio: float = 0.3,
    num_workers: int = 1,
) -> dict:
    """Slice every split under ``{source_images_dir}/{split}/``.

    Output layout::

        {output_dir}/images/{split}/{stem}_{x1}_{y1}_{x2}_{y2}{ext}
        {output_dir}/labels/{split}/{stem}_{x1}_{y1}_{x2}_{y2}.txt

    Returns a dict keyed by split name.
    """
    source_images_dir = Path(source_images_dir)
    source_labels_dir = Path(source_labels_dir)
    output_dir = Path(output_dir)

    results: dict = {}
    for split in splits:
        in_imgs = source_images_dir / split
        in_lbls = source_labels_dir / split
        if not in_imgs.is_dir() or not in_lbls.is_dir():
            logger.warning("Skipping split %r: missing %s or %s", split, in_imgs, in_lbls)
            continue
        results[split] = slice_split(
            images_dir=in_imgs,
            labels_dir=in_lbls,
            output_images_dir=output_dir / "images" / split,
            output_labels_dir=output_dir / "labels" / split,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            min_area_ratio=min_area_ratio,
            num_workers=num_workers,
        )

    return results


def write_sliced_dataset_yaml(
    output_dir: Path | str,
    class_names: dict[int, str],
) -> Path:
    """Write a YOLO ``dataset.yaml`` for the sliced dataset.

    Assumes the standard ``images/{train,val,test}`` layout.
    """
    output_dir = Path(output_dir)
    yaml_path = output_dir / "dataset.yaml"
    names_block = "\n".join(f"  {cid}: {name}" for cid, name in sorted(class_names.items()))
    yaml_path.write_text(
        f"# This dataset.yaml is portable — it resolves relative to *this file's* directory.\n"
        f"# Sync the parent directory to any machine (Colab, inference server) and train in place.\n"
        f"path: .\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"\n"
        f"names:\n"
        f"{names_block}\n",
        encoding="utf-8",
    )
    logger.info("Wrote dataset.yaml → %s", yaml_path)
    return yaml_path
