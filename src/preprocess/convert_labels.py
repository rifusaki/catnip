"""
Convert Label Studio per-annotation JSON exports to YOLO format.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def _default_stage1_remap(label_name: str) -> str:
    """Remap any _body/_face label to body or face (class-agnostic for Stage 1)."""
    if label_name.endswith("_body"):
        return "body"
    if label_name.endswith("_face"):
        return "face"
    return label_name


def convert_annotations_directory(
    dir_path: str,
    output_dir: str,
    class_map: dict | None = None,
    remap_fn=None,
):
    """
    Reads per-annotation JSON files from a directory and converts to YOLO format.

    New Label Studio per-annotation export format (one file per annotation):
        {
          "id": 31,
          "result": [
            {
              "value": {
                "x": 51.8, "y": 1.12, "width": 34.7, "height": 72.6,
                "rectanglelabels": ["izutsumi_body"]
              },
              ...
            }
          ],
          "task": {
            "data": {"image": "gs://catnip-data/data/manga/v09/0043.jpg"}
          }
        }

    Key differences from the old bulk-export format:
     - One file per annotation (not all tasks in one JSON array).
     - Image URL at ``task.data.image``.
     - URL contains ``/data/manga/`` prefix.
     - Annotations at top-level ``result[]`` (not ``annotations[].result[]``).

    Groups annotations by image path so that multiple annotation files
    referencing the same image accumulate lines in a single .txt file.

    Args:
        dir_path: Directory containing per-annotation JSON files.
        output_dir: Root directory to save YOLO .txt label files.
        class_map: Dict mapping (remapped) label names → class IDs.
        remap_fn: Optional function ``(str → str)`` to transform label
                  names before ``class_map`` lookup.
    """
    if class_map is None:
        try:
            from src.config import settings
            class_map = settings.labels.stage1.model_dump()
        except (ImportError, AttributeError):
            class_map = {"body": 0, "face": 1}

    if remap_fn is None:
        remap_fn = _default_stage1_remap

    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        print(f"Error: Directory not found: '{dir_path}'")
        sys.exit(1)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[str]] = defaultdict(list)
    file_count = 0
    skipped = 0

    for entry in sorted(dir_path.iterdir()):
        if not entry.is_file():
            continue

        try:
            with open(entry, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Warning: Skipping {entry.name}: {exc}")
            skipped += 1
            continue

        file_count += 1

        task = data.get('task', {})
        task_data = task.get('data', {})
        url = task_data.get('image', '')

        if not url:
            print(f"Warning: Skipping {entry.name}: No 'task.data.image' URL")
            skipped += 1
            continue

        if "manga/" not in url:
            print(f"Warning: Skipping {entry.name}: Unexpected URL format '{url}'")
            skipped += 1
            continue

        rel_path = url.split("manga/")[-1]

        results = data.get('result', [])
        if not results:
            continue

        yolo_lines = []
        for result in results:
            if result.get('type') != 'rectanglelabels':
                continue

            value = result.get('value')
            if not value:
                continue

            labels = value.get('rectanglelabels', [])
            if not labels:
                continue

            original_label = labels[0]
            label_name = remap_fn(original_label)

            if label_name not in class_map:
                print(
                    f"Warning: Unknown label '{label_name}' "
                    f"(from '{original_label}') in {rel_path}. Skipping."
                )
                continue

            class_id = class_map[label_name]

            x = value['x']
            y = value['y']
            w = value['width']
            h = value['height']

            # Label Studio (0-100) → YOLO (0-1), top-left → centre
            x_center = (x + w / 2) / 100.0
            y_center = (y + h / 2) / 100.0
            w_norm = w / 100.0
            h_norm = h / 100.0

            yolo_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} "
                f"{w_norm:.6f} {h_norm:.6f}"
            )

        if yolo_lines:
            grouped[rel_path].extend(yolo_lines)

    # Write one .txt file per unique source image
    written = 0
    for rel_path, lines in grouped.items():
        image_rel_path = Path(rel_path)
        label_rel_path = image_rel_path.with_suffix('.txt')
        label_full_path = output_dir / label_rel_path
        label_full_path.parent.mkdir(parents=True, exist_ok=True)

        with open(label_full_path, 'w') as f:
            f.write('\n'.join(lines))

        written += 1

    print(
        f"Processed {file_count} annotation files ({skipped} skipped). "
        f"Wrote {written} label files to {output_dir}"
    )
