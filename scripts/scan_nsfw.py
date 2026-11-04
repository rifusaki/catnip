#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch>=2.4,<3",
#     "transformers>=4.44,<5",
#     "pillow>=10,<12",
#     "tqdm>=4.66,<5",
# ]
# ///
"""
scan_nsfw.py — Score every image in a YOLO dataset with an NSFW classifier and
report which sources/titles trip it, so they can be excluded before uploading
to Kaggle (which auto-unpublishes anything its classifier deems NSFW).

Uses the ``Falconsai/nsfw_image_detection`` ViT classifier via ``transformers``.
For each image it records the predicted ``nsfw`` score; any image at or above
``--threshold`` is flagged. The report aggregates flagged images by source
prefix and (for manga109) by title, and prints a ready-to-paste
``--exclude-title`` / ``--exclude-source`` list for ``sanitize_names.py``.

This script is self-contained: it runs via ``uv`` (PEP 723 inline deps) in an
ephemeral venv, so it does not touch the pixi training env. torch on macOS
arm64 ships MPS support out of the box. First run downloads torch + transformers
(~100 MB) plus the model (~150 MB).

Usage::

    uv run scripts/scan_nsfw.py --input-dir /Volumes/rifuSSD/catnip-data/training/stage1_sliced
    uv run scripts/scan_nsfw.py --input-dir ... --limit 200 --device mps   # quick smoke test
    uv run scripts/scan_nsfw.py --input-dir ... --threshold 0.5 --output nsfw_report.json

    # then feed the printed exclusions back into the pipeline:
    pixi run sanitize-names --exclude-title <T1> --exclude-title <T2> ...
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

logger = logging.getLogger("scan_nsfw")

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})
MODEL_ID = "Falconsai/nsfw_image_detection"


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

def source_of(stem: str) -> str:
    """Leading ``_``-delimited token — the corpus a file came from."""
    return stem.split("_", 1)[0]


def title_of(stem: str) -> str | None:
    """Manga109 title (the second ``_``-token), or None for non-manga109."""
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0] == "manga109":
        return parts[1]
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flag NSFW images so they can be excluded from the Kaggle upload.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("catnip-data/training/stage1_sliced"),
        help=(
            "YOLO dataset root to scan (default: catnip-data/training/stage1_sliced)."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="NSFW score at/above which an image is flagged (default: 0.5).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto (default), mps, cpu, cuda.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Images per classifier batch (default: 32).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Scan only the first N images (0 = all; default: 0).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("nsfw_report.json"),
        help="Where to write the JSON report (default: nsfw_report.json).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _collect_images(input_dir: Path) -> list[Path]:
    images_dir = input_dir / "images"
    if not images_dir.is_dir():
        logger.error("images/ not found under %s", input_dir)
        sys.exit(1)
    images = sorted(
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        logger.error("No images found under %s", images_dir)
        sys.exit(1)
    return images


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(message)s",
    )

    if not (0.0 <= args.threshold <= 1.0):
        logger.error("--threshold must be in [0, 1], got %s", args.threshold)
        sys.exit(1)

    images = _collect_images(args.input_dir)
    if args.limit > 0:
        images = images[: args.limit]
    total = len(images)
    logger.info("Scanning %d images under %s", total, args.input_dir)

    device = _resolve_device(args.device)
    logger.info("Loading %s on %s (first run downloads ~150 MB) ...", MODEL_ID, device)

    from transformers import pipeline
    from tqdm import tqdm

    classifier = pipeline(
        "image-classification",
        model=MODEL_ID,
        device=device,
    )

    flagged: list[dict] = []
    by_source: Counter = Counter()
    by_title: Counter = Counter()

    batch_size = max(1, args.batch_size)
    for i in tqdm(range(0, total, batch_size), desc="scanning", unit="batch"):
        batch = images[i : i + batch_size]
        results = classifier([str(p) for p in batch], top_k=2)
        for path, result in zip(batch, results):
            scores = {r["label"].lower(): r["score"] for r in result}
            nsfw_score = scores.get("nsfw", 0.0)
            if nsfw_score < args.threshold:
                continue
            stem = path.stem
            src = source_of(stem)
            title = title_of(stem)
            by_source[src] += 1
            if title:
                by_title[(src, title)] += 1
            flagged.append(
                {
                    "path": str(path),
                    "source": src,
                    "title": title,
                    "score": round(nsfw_score, 4),
                }
            )

    report = {
        "model": MODEL_ID,
        "threshold": args.threshold,
        "scanned": total,
        "flagged": len(flagged),
        "by_source": dict(by_source.most_common()),
        "by_title": {f"{s}/{t}": n for (s, t), n in by_title.most_common()},
        "flagged_images": flagged,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # --- Human-readable summary + ready-to-paste exclusions ---
    print(f"\n{'=' * 60}\n{' NSFW Scan Complete '.center(60, '=')}\n{'=' * 60}")
    print(f"  Scanned:  {total}")
    print(f"  Flagged:  {len(flagged)}  (threshold >= {args.threshold})")
    print(f"  Report:   {args.output}")

    if not flagged:
        print("\n  No images flagged. Nothing to exclude.")
        print(f"{'=' * 60}\n")
        return

    print(f"\n  Flagged by source:")
    for src, n in by_source.most_common():
        print(f"    {src:20s} {n:>7}")

    if by_title:
        print(f"\n  Flagged manga109 titles:")
        for (src, title), n in by_title.most_common():
            print(f"    {title:30s} {n:>7}")

    print("\n  Suggested sanitize exclusions:")
    for (src, title), _ in by_title.most_common():
        print(f"    --exclude-title {title}")
    for src in sorted(set(by_source) - {"manga109"}):
        print(f"    --exclude-source {src}   # {by_source[src]} flagged")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
