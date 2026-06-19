"""
Manga109 XML annotation parser for Stage 1 YOLO format.

Extracts ``<body>`` and ``<face>`` bounding-box annotations from per-title
XML files, normalises pixel coordinates to YOLO format, and returns a
parsed structure ready for image copy + label write in the CLI wrapper.

Manga109 provides no ``<head>`` annotations — the head class is filled in
later by the YOLO head-detection datasets during unification.
"""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Stage 1 class mapping (spec-compliant: body=0, head=1, face=2)
CLASS_MAP: dict[str, int] = {
    "body": 0,
    "head": 1,
    "face": 2,
}

# Manga109 element tags we extract — everything else (text, frame) is ignored
RELEVANT_TAGS: frozenset[str] = frozenset({"body", "face"})

# Number of YOLO label fields per line
LABEL_FIELDS: int = 5  # class_id x_center y_center w h

# Image naming: manga109_{title}_{index:03d}.jpg
DEST_IMAGE_TEMPLATE: str = "manga109_{title}_{index:03d}.jpg"


# ---------------------------------------------------------------------------
# Parse XML annotations → per-page lists of YOLO-format lines (Law 2)
# ---------------------------------------------------------------------------

def float_attr(element: ET.Element, name: str) -> float:
    """Read a required float attribute, raising on failure (Law 4 — Fail Fast)."""
    value = element.get(name)
    if value is None:
        raise ValueError(f"Missing required attribute '{name}' on <{element.tag}>")
    return float(value)


def parse_xml_annotations(
    xml_path: Path,
) -> tuple[str, dict[int, tuple[list[str], int, int]], int]:
    """Parse one Manga109 XML file.

    Returns ``(title, pages, skipped_oob)`` where *pages* maps
    ``page_index → (label_lines, width, height)`` and *skipped_oob* is the
    count of annotations whose normalised coordinates fell outside [0,1].
    Pages with no body/face annotations are excluded from the result.
    """
    tree = ET.parse(xml_path)
    book = tree.getroot()
    title = book.attrib["title"]

    pages: dict[int, tuple[list[str], int, int]] = {}
    skipped_oob = 0

    for page_elem in book.findall("pages/page"):
        index_str = page_elem.get("index")
        width_str = page_elem.get("width")
        height_str = page_elem.get("height")

        # Guard: missing page dimensions (Law 4 — Fail Fast)
        if index_str is None or width_str is None or height_str is None:
            raise ValueError(
                f"<page> in '{title}' missing index/width/height: {page_elem.attrib!r}"
            )

        page_index = int(index_str)
        page_width = float(width_str)
        page_height = float(height_str)

        label_lines: list[str] = []

        for child in page_elem:
            tag = child.tag
            if tag not in RELEVANT_TAGS:
                continue

            class_id = CLASS_MAP[tag]

            # Parse pixel coordinates (Law 2 — parse at boundary)
            xmin = float_attr(child, "xmin")
            ymin = float_attr(child, "ymin")
            xmax = float_attr(child, "xmax")
            ymax = float_attr(child, "ymax")

            # Convert to YOLO normalised format
            x_center = (xmin + xmax) / (2.0 * page_width)
            y_center = (ymin + ymax) / (2.0 * page_height)
            w_norm = (xmax - xmin) / page_width
            h_norm = (ymax - ymin) / page_height

            # Guard: skip out-of-bounds or degenerate boxes (Law 4 — Fail Fast)
            if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
                skipped_oob += 1
                logger.warning(
                    "  Skipping OOB annotation in '%s' page %d: "
                    "center (%.4f, %.4f) outside [0,1]",
                    title, page_index, x_center, y_center,
                )
                continue
            if w_norm > 1.0 or h_norm > 1.0:
                skipped_oob += 1
                logger.warning(
                    "  Skipping OOB annotation in '%s' page %d: "
                    "size (%.4f, %.4f) exceeds 1.0",
                    title, page_index, w_norm, h_norm,
                )
                continue
            if w_norm <= 0.0 or h_norm <= 0.0:
                skipped_oob += 1
                logger.warning(
                    "  Skipping degenerate annotation in '%s' page %d: "
                    "non-positive size (%.4f, %.4f)",
                    title, page_index, w_norm, h_norm,
                )
                continue

            label_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
            )

        # Guard: skip pages with no annotations (Law 1 — Early Exit)
        if not label_lines:
            continue

        pages[page_index] = (label_lines, int(page_width), int(page_height))

    return title, pages, skipped_oob


# ---------------------------------------------------------------------------
# Image / label output (Law 3 — Atomic Predictability)
# ---------------------------------------------------------------------------

def process_title(
    title: str,
    pages: dict[int, tuple[list[str], int, int]],
    images_dir: Path,
    images_out: Path,
    labels_out: Path,
    skipped_oob: int,
) -> dict:
    """Copy images and write YOLO labels for one manga title.

    Returns a summary dict with keys: pages_processed, body_count, face_count,
    head_count, skipped_oob, images_copied.
    """
    import shutil  # local import — only needed when I/O is performed

    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    title_images_dir = images_dir / title

    # Guard: images directory missing (Law 1 — Early Exit)
    if not title_images_dir.is_dir():
        logger.warning(
            "  Images directory not found for '%s': %s — skipping",
            title,
            title_images_dir,
        )
        return {
            "pages_processed": 0,
            "body_count": 0,
            "face_count": 0,
            "head_count": 0,
            "skipped_oob": skipped_oob,
            "images_copied": 0,
        }

    pages_processed = 0
    body_count = 0
    face_count = 0
    images_copied = 0

    for page_index, (label_lines, _width, _height) in sorted(pages.items()):
        src_image = title_images_dir / f"{page_index:03d}.jpg"

        # Guard: image missing (Law 1 — skip with warning)
        if not src_image.is_file():
            logger.warning(
                "  Image not found: %s — skipping page %d of '%s'",
                src_image,
                page_index,
                title,
            )
            continue

        dest_name = DEST_IMAGE_TEMPLATE.format(title=title, index=page_index)
        dest_image = images_out / dest_name
        dest_label = labels_out / f"{dest_image.stem}.txt"

        shutil.copy2(src_image, dest_image)
        images_copied += 1

        dest_label.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
        pages_processed += 1

        for line in label_lines:
            class_id = int(line.split()[0])
            if class_id == CLASS_MAP["body"]:
                body_count += 1
            elif class_id == CLASS_MAP["face"]:
                face_count += 1

    return {
        "pages_processed": pages_processed,
        "body_count": body_count,
        "face_count": face_count,
        "head_count": 0,
        "skipped_oob": skipped_oob,
        "images_copied": images_copied,
    }
