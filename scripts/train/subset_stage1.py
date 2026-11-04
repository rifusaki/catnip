#!/usr/bin/env python3
"""
subset_stage1.py — Build a stratified symlink subset of stage1_sliced.

The full SAHI-sliced dataset (~165k slices) is far too large for local
MPS training: most slices are redundant — overlapping manga page slices
(0.2 overlap) and near-duplicate single-object head crops. This script
keeps the *pixel distribution* (imgsz stays 640) while cutting redundancy:

  * per-source stratification — head(1) boxes exist only in ah_yolo /
    ah_coco, body(0)/face(2) only in manga109/izutsumi, so every source
    must stay represented (unmapped sources abort the build);
  * max_per_image caps the slices kept per original image — manga109
    expanded 7.3x under slicing, so one slice per page keeps every page
    in the subset at 1/9th the cost;
  * fraction is a per-slice keep-probability for the long-tail sources.

Output: <unified_dir>_subset/ (sibling of stage1_sliced) with real
images/{train,val} + labels/{train,val} dirs of *relative* symlinks, a
dataset.yaml, and subset_manifest.json (knobs + per-source counts).
Test split is NOT subset — run final metrics on the full test split.

Usage::

    python scripts/train/subset_stage1.py            # build with config knobs
    python scripts/train/subset_stage1.py --dry-run  # stats only
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import shutil
import sys
import os
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import settings

logger = logging.getLogger("subset_stage1")

IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})
# Slice files are named {original_stem}_{minx}_{miny}_{w}_{h}{ext}.
SLICE_COORDS_RE = re.compile(r"^(.*)_(\d+)_(\d+)_(\d+)_(\d+)$")
MANIFEST_NAME = "subset_manifest.json"
SOURCES = ("izutsumi", "manga109", "ah_yolo", "ah_coco")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a stratified symlink subset of stage1_sliced for local training.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview sampling stats only — no files written.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug-level logging.",
    )
    return parser.parse_args()


def build_stem_to_source(staging_root: Path) -> dict[str, str]:
    """Map every staging image stem to its source name; fail loud if incomplete."""
    if not staging_root.is_dir():
        logger.error("Staging root not found: %s. Re-run the unify-* tasks.", staging_root)
        sys.exit(1)

    stem_to_source: dict[str, str] = {}
    missing = [s for s in SOURCES if not (staging_root / s / "images").is_dir()]
    if missing:
        logger.error(
            "Staging dirs missing for %s — source mapping is impossible. "
            "Re-run: pixi run data-prep (or the individual unify-* tasks).",
            ", ".join(missing),
        )
        sys.exit(1)

    for source in SOURCES:
        for f in (staging_root / source / "images").iterdir():
            if f.suffix.lower() in IMAGE_EXTS:
                stem_to_source[f.stem] = source
    return stem_to_source


def original_stem(slice_stem: str) -> str:
    """Strip the _{minx}_{miny}_{w}_{h} suffix; unsliced names pass through."""
    m = SLICE_COORDS_RE.match(slice_stem)
    return m.group(1) if m else slice_stem


def sample_split(
    images_dir: Path,
    labels_dir: Path,
    stem_to_source: dict[str, str],
    knobs: dict,
    rng,
) -> tuple[list[Path], dict]:
    """Apply per-source {fraction, max_per_image} sampling to one split.

    Returns (kept image paths sorted, per-source stats).
    """
    unknown_sources: set[str] = set()
    by_image: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for img in images_dir.iterdir():
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        source = stem_to_source.get(original_stem(img.stem))
        if source is None:
            logger.error("Slice %s has no staging provenance — aborting.", img.name)
            sys.exit(1)
        if source not in knobs:
            unknown_sources.add(source)
        by_image[(source, original_stem(img.stem))].append(img)

    if unknown_sources:
        logger.error(
            "Config training.stage1.subset is missing sources: %s. "
            "A missing source would starve its classes (head only lives in ah_*).",
            ", ".join(sorted(unknown_sources)),
        )
        sys.exit(1)

    stats: dict[str, dict] = {}
    kept: list[Path] = []
    for source in sorted(knobs):
        fraction = knobs[source].fraction
        max_per_image = knobs[source].max_per_image
        slices_of_source = [
            slices for (src, _), slices in by_image.items() if src == source
        ]
        total_slices = sum(len(s) for s in slices_of_source)
        source_kept: list[Path] = []
        for slices in slices_of_source:
            chosen = slices if max_per_image <= 0 else rng.sample(slices, min(max_per_image, len(slices)))
            source_kept.extend(p for p in chosen if fraction >= 1.0 or rng.random() < fraction)
        kept.extend(source_kept)
        stats[source] = {"images": len(source_kept), "from": total_slices}

    return sorted(kept), stats


def label_stats(kept: list[Path], labels_dir: Path) -> tuple[int, Counter]:
    """Count backgrounds (empty labels) and class occurrences for kept slices."""
    backgrounds = 0
    classes: Counter = Counter()
    for img in kept:
        label = labels_dir / f"{img.stem}.txt"
        if not label.is_file():
            logger.error("Missing label for kept slice %s — aborting.", img.name)
            sys.exit(1)
        if label.stat().st_size == 0:
            backgrounds += 1
            continue
        for line in label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts:
                classes[int(parts[0])] += 1
    return backgrounds, classes


def prepare_output(root: Path) -> None:
    """Wipe a previous subset build; refuse to touch anything we didn't create."""
    if root.exists():
        if not root.is_dir():
            logger.error("Refusing: %s exists and is not a directory.", root)
            sys.exit(1)
        is_ours = (root / MANIFEST_NAME).is_file() or not any(root.iterdir())
        if not is_ours:
            logger.error(
                "Refusing to wipe %s: no %s inside. Remove it manually if intended.",
                root, MANIFEST_NAME,
            )
            sys.exit(1)
        shutil.rmtree(root)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)


def link_split(kept: list[Path], sliced_root: Path, split: str, root: Path) -> None:
    """Create relative symlinks for images and labels of one split."""
    linked = 0
    for img in kept:
        img_link = root / "images" / split / img.name
        os.symlink(os.path.relpath(img, img_link.parent), img_link)
        label = sliced_root / "labels" / split / f"{img.stem}.txt"
        label_link = root / "labels" / split / label.name
        os.symlink(os.path.relpath(label, label_link.parent), label_link)
        linked += 1
    logger.info("[%s] linked %d image+label pairs", split, linked)


def write_dataset_yaml(root: Path, class_names: dict[int, str]) -> None:
    names_block = "\n".join(f"  {cid}: {name}" for cid, name in sorted(class_names.items()))
    (root / "dataset.yaml").write_text(
        f"# Stratified subset of the SAHI-sliced stage1 dataset (symlinks).\n"
        f"# Built by scripts/train/subset_stage1.py — see subset_manifest.json.\n"
        f"path: {root.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"names:\n"
        f"{names_block}\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(message)s",
    )

    subset_cfg = settings.training.stage1.subset
    sliced_root = settings.paths.stage1_sliced_dir
    staging_root = settings.paths.data / "data" / "staging"
    subset_root = settings.paths.stage1_sliced_subset_dir
    class_names = {v: k for k, v in settings.labels.stage1.model_dump().items()}

    if not (sliced_root / "dataset.yaml").is_file():
        logger.error("Sliced dataset not found at %s. Run `pixi run unify-stage1` first.", sliced_root)
        sys.exit(1)

    stem_to_source = build_stem_to_source(staging_root)
    rng = random.Random(subset_cfg.seed)

    manifest: dict = {"seed": subset_cfg.seed, "sliced_root": str(sliced_root), "splits": {}}
    per_split_kept: dict[str, list[Path]] = {}
    for split in ("train", "val"):
        knobs = subset_cfg.train if split == "train" else subset_cfg.val
        kept, stats = sample_split(
            sliced_root / "images" / split, sliced_root / "labels" / split,
            stem_to_source, knobs, rng,
        )
        backgrounds, classes = label_stats(kept, sliced_root / "labels" / split)
        manifest["splits"][split] = {
            "total": len(kept),
            "backgrounds": backgrounds,
            "classes": {str(cid): n for cid, n in sorted(classes.items())},
            "per_source": stats,
        }
        per_split_kept[split] = kept
        logger.info(
            "[%s] kept %d slices (%d background), classes: %s",
            split, len(kept), backgrounds, dict(classes),
        )

    if args.dry_run:
        logger.info("=== DRY RUN — nothing written ===")
        return

    prepare_output(subset_root)
    for split in ("train", "val"):
        link_split(per_split_kept[split], sliced_root, split, subset_root)
    write_dataset_yaml(subset_root, class_names)
    (subset_root / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )

    train_total = manifest["splits"]["train"]["total"]
    val_total = manifest["splits"]["val"]["total"]
    logger.info(
        "Subset ready: %d train / %d val → %s (test split NOT subset — "
        "run final metrics on the full test split)",
        train_total, val_total, subset_root,
    )


if __name__ == "__main__":
    main()
