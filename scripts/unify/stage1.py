#!/usr/bin/env python3
"""
unify_stage1.py — Merge 4 dataset staging directories, shuffle/split into
train/val/test, and generate dataset.yaml for YOLO26 Stage 1 training.

Assumes staging directories already exist from prior runs of individual
conversion scripts.  Does **not** invoke subprocesses.

Usage::

    python scripts/unify/stage1.py
    python scripts/unify/stage1.py --split 80 10 10
    python scripts/unify/stage1.py --dry-run
    python scripts/unify/stage1.py --output-dir /custom/path
    python scripts/unify/stage1.py --seed 42 --verbose
    python scripts/unify/stage1.py --slice --slice-workers 4

Paths are resolved from config/pipeline.yaml. Set CATNIP_DATA env var to
relocate the entire catnip-data tree (e.g. to an external SSD or gcsfuse mount).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

# Allow running the script directly from scripts/unify/ without an installed package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from src.config import settings

STAGING_DIRS: list[Path] = [
    settings.paths.data / "data" / "staging" / name
    for name in ("izutsumi", "manga109", "ah_yolo", "ah_coco")
]

DEFAULT_OUTPUT = settings.paths.unified_dir

CLASS_NAMES: dict[int, str] = {0: "body", 1: "head", 2: "face"}
IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})
MIN_REQUIRED_PAIRS = 10

logger = logging.getLogger("unify_stage1")


# ========================================================================
# CLI  (Law 2 — Parse at the boundary)
# ========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge staging dirs, split, and generate YOLO dataset.yaml."
    )
    parser.add_argument(
        "--split", nargs=3, type=int, default=[80, 10, 10],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Split ratios (must sum to 100). Default: 80 10 10.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT,
        help="Final output directory (default from config: paths.unified_dir).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview stats only — no files written.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible splits (default: 42).",
    )
    parser.add_argument(
        "--slice", action="store_true",
        help="After unification, pre-slice every image using the SAHI "
             "parameters from config/pipeline.yaml (params.sahi.*). "
             "Output is written to <output-dir>_sliced/. Gives the model "
             "train/inference parity with SAHI sliced prediction.",
    )
    parser.add_argument(
        "--slice-workers", type=int, default=1,
        help="Number of parallel processes for slicing (default: 1 = serial).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug-level logging.",
    )
    args = parser.parse_args()

    if sum(args.split) != 100:
        parser.error(f"--split must sum to 100, got {args.split} (sum={sum(args.split)})")
    if any(r <= 0 for r in args.split):
        parser.error("All --split ratios must be positive integers")

    return args


# ========================================================================
# Parsing helpers  (Law 2 — Parse at the boundary)
# ========================================================================

def _read_label_classes(label_path: Path) -> list[int]:
    """Return class IDs from a YOLO label file, skipping malformed lines."""
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        logger.warning("Cannot read label: %s", label_path)
        return []
    ids: list[int] = []
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if parts:
            try:
                ids.append(int(parts[0]))
            except ValueError:
                logger.warning("Malformed class in %s: %r", label_path, stripped)
    return ids


def _find_image(stem: str, images_dir: Path) -> Path | None:
    """Return the image file matching *stem* in *images_dir*, or None."""
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


# ========================================================================
# Collection
# ========================================================================

def collect_pairs(
    staging_dirs: list[Path],
) -> tuple[list[tuple[Path, Path, str]], dict[str, Counter], int]:
    """Walk staging dirs collecting (image_path, label_path, source) triples.

    Returns:
        all_pairs: list of (image_path, label_path, source_name)
        class_dist_by_source: source_name → Counter[class_id → count]
    """
    all_pairs: list[tuple[Path, Path, str]] = []
    class_dist_by_source: dict[str, Counter] = {}
    seen_stems: dict[str, str] = {}
    duplicates_excluded = 0

    for staging in staging_dirs:
        src = staging.name
        im_dir, lb_dir = staging / "images", staging / "labels"

        # Law 1 — skip missing/incomplete dirs gracefully
        if not staging.is_dir():
            logger.warning("[%s] Staging dir not found — skipping.", src)
            class_dist_by_source[src] = Counter()
            continue
        if not im_dir.is_dir() or not lb_dir.is_dir():
            logger.warning("[%s] Missing images/ or labels/ — skipping.", src)
            class_dist_by_source[src] = Counter()
            continue

        label_files = sorted(lb_dir.glob("*.txt"))
        if not label_files:
            logger.warning("[%s] No label files — skipping.", src)
            class_dist_by_source[src] = Counter()
            continue

        counter: Counter = Counter()
        src_pairs: list[tuple[Path, Path]] = []

        for lb in label_files:
            im = _find_image(lb.stem, im_dir)
            if im is None:
                logger.warning("[%s] No image for %s — skipping.", src, lb.name)
                continue
            for cid in _read_label_classes(lb):
                counter[cid] += 1
            src_pairs.append((im, lb))

        deduped_pairs: list[tuple[Path, Path]] = []
        for im, lb in src_pairs:
            if im.stem in seen_stems:
                logger.warning("Duplicate stem '%s': in '%s' and '%s'.", im.stem, seen_stems[im.stem], src)
                duplicates_excluded += 1
            else:
                seen_stems[im.stem] = src
                deduped_pairs.append((im, lb))

        all_pairs.extend((im, lb, src) for im, lb in deduped_pairs)
        class_dist_by_source[src] = counter
        logger.info("[%s] %d pairs, %d annotations.", src, len(src_pairs), sum(counter.values()))

    return all_pairs, class_dist_by_source, duplicates_excluded


# ========================================================================
# Shuffle & split  (Law 3 — Atomic Predictability)
# ========================================================================

def shuffle_and_split(
    all_pairs: list[tuple[Path, Path, str]],
    train_pct: int,
    val_pct: int,
    seed: int,
) -> tuple[
    list[tuple[Path, Path, str]],
    list[tuple[Path, Path, str]],
    list[tuple[Path, Path, str]],
]:
    total = len(all_pairs)
    if total < MIN_REQUIRED_PAIRS:
        logger.error("Insufficient data: need >= %d pairs, got %d.", MIN_REQUIRED_PAIRS, total)
        sys.exit(1)

    shuffled = list(all_pairs)
    random.Random(seed).shuffle(shuffled)

    n_train = total * train_pct // 100
    n_val = total * val_pct // 100

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]

    for label, subset in [("train", train), ("val", val), ("test", test)]:
        if not subset:
            logger.error("Split '%s' is empty (%d total pairs). Adjust --split.", label, total)
            sys.exit(1)

    return train, val, test


# ========================================================================
# File output
# ========================================================================

def _prepare_output(output_root: Path) -> tuple[Path, ...]:
    """Create nested images/{train,val,test} and labels/{train,val,test}."""
    if output_root.exists():
        shutil.rmtree(output_root)
    subdirs = []
    for kind in ("images", "labels"):
        for split_name in ("train", "val", "test"):
            d = output_root / kind / split_name
            d.mkdir(parents=True)
            subdirs.append(d)
    return tuple(subdirs)


def _compute_stats(pairs: list[tuple[Path, Path, str]]) -> tuple[int, Counter]:
    """Return (image_count, class_distribution) for dry-run stats."""
    c: Counter = Counter()
    for _, lb, _ in pairs:
        for cid in _read_label_classes(lb):
            c[cid] += 1
    return len(pairs), c


def copy_split(
    pairs: list[tuple[Path, Path, str]],
    im_dst: Path,
    lb_dst: Path,
) -> tuple[int, Counter]:
    """Copy files for one split. Returns (count, class_distribution)."""
    c: Counter = Counter()
    n = 0
    for im, lb, _ in pairs:
        shutil.copy2(im, im_dst / im.name)
        shutil.copy2(lb, lb_dst / lb.name)
        n += 1
        for cid in _read_label_classes(lb):
            c[cid] += 1
    return n, c


# ========================================================================
# Metadata
# ========================================================================

def _write_dataset_yaml(output_root: Path) -> None:
    path = output_root / "dataset.yaml"
    path.write_text(
        f"path: {output_root.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"\n"
        f"names:\n"
        f"  0: body\n"
        f"  1: head\n"
        f"  2: face\n",
        encoding="utf-8",
    )
    logger.info("Wrote dataset.yaml → %s", path)


def _write_split_manifest(per_source: dict[str, dict[str, int]], output_root: Path) -> None:
    path = output_root / "split_manifest.json"
    path.write_text(json.dumps(per_source, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Wrote split_manifest.json → %s", path)


# ========================================================================
# Summary  (Law 5 — Intentional Naming)
# ========================================================================

def _print_summary(
    class_dist_by_source: dict[str, Counter],
    per_source_counts: dict[str, dict[str, int]],
    train_c: Counter,
    val_c: Counter,
    test_c: Counter,
    total_imgs: int,
    duplicates_excluded: int,
    output_root: Path,
) -> None:
    total_anns = sum(sum(c.values()) for c in (train_c, val_c, test_c))
    name = lambda cid: CLASS_NAMES.get(cid, f"?{cid}")

    print(f"\n{'=' * 60}\n{' Stage 1 Dataset Unification — Complete '.center(60, '=')}\n{'=' * 60}")

    # Per source
    print(f"\n  Per-Source Contributions")
    g_imgs = g_anns = 0
    for src in sorted(per_source_counts):
        c = class_dist_by_source.get(src, Counter())
        n_i = sum(per_source_counts[src].values())
        n_a = sum(c.values())
        g_imgs += n_i; g_anns += n_a
        print(f"  {src:18s}  images={n_i:>6}  annotations={n_a:>6}")
        for cid in sorted(c):
            print(f"    {'':18s}  {name(cid):6s} (class {cid}): {c[cid]:>6}")
    print(f"  {'TOTAL':18s}  images={g_imgs:>6}  annotations={g_anns:>6}")

    # Per split
    print(f"\n  Per-Split Distribution")
    for sname, c in [("train", train_c), ("val", val_c), ("test", test_c)]:
        s_imgs = sum(per_source_counts.get(src, {}).get(sname, 0) for src in per_source_counts)
        pct = (s_imgs / total_imgs * 100) if total_imgs else 0
        print(f"  {sname:8s}  images: {s_imgs:>6} ({pct:5.1f}%)  annotations: {sum(c.values()):>6}")
        for cid in sorted(c):
            print(f"    {'':8s}  {name(cid):6s} (class {cid}): {c[cid]:>6}")

    # Overall
    print(f"\n  Overall")
    print(f"  Total images:        {total_imgs:>6}")
    print(f"  Total annotations:   {total_anns:>6}")
    if duplicates_excluded > 0:
        print(f"  Duplicates excluded: {duplicates_excluded:>6}")
    if total_anns > 0:
        print(f"  Class distribution:")
        for cid in sorted(CLASS_NAMES):
            t = train_c.get(cid, 0) + val_c.get(cid, 0) + test_c.get(cid, 0)
            print(f"    {name(cid):6s} (class {cid}): {t:>6} ({t / total_anns * 100:5.1f}%)")
    print(f"\n  Output: {output_root.resolve()}\n{'=' * 60}\n")


# ========================================================================
# Main
# ========================================================================

def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(message)s",
    )

    if args.dry_run:
        logger.info("=== DRY RUN — no files will be written ===")
    logger.info("Project root: %s  |  seed: %d", PROJECT_ROOT, args.seed)

    # Phase 1 — Collect pairs
    logger.info("Collecting image-label pairs from staging directories …")
    all_pairs, class_dist_by_source, duplicates_excluded = collect_pairs(STAGING_DIRS)
    if not all_pairs:
        logger.error("No pairs found in any staging directory. Run conversion scripts first.")
        sys.exit(1)

    # Phase 2 — Shuffle & split
    tp, vp, xp = args.split
    logger.info("Shuffling & splitting %d pairs (%d/%d/%d) …", len(all_pairs), tp, vp, xp)
    train, val, test = shuffle_and_split(all_pairs, tp, vp, args.seed)

    # Phase 3 — Copy (or dry-run stats)
    output = args.output_dir
    if args.dry_run:
        logger.info("--- DRY RUN: would output to %s ---", output)
        _, train_c = _compute_stats(train)
        _, val_c = _compute_stats(val)
        _, test_c = _compute_stats(test)
    else:
        logger.info("Preparing output: %s", output)
        im_tr, im_va, im_te, lb_tr, lb_va, lb_te = _prepare_output(output)
        n_tr, train_c = copy_split(train, im_tr, lb_tr)
        n_va, val_c = copy_split(val, im_va, lb_va)
        n_te, test_c = copy_split(test, im_te, lb_te)
        logger.info("Copied %d train / %d val / %d test images.", n_tr, n_va, n_te)
        _write_dataset_yaml(output)

    # Phase 4 — Per-source per-split counts
    src_names = sorted({src for _, _, src in all_pairs})
    per_source = {n: {"train": 0, "val": 0, "test": 0} for n in src_names}
    for _, _, src in train: per_source[src]["train"] += 1
    for _, _, src in val:   per_source[src]["val"]   += 1
    for _, _, src in test:  per_source[src]["test"]  += 1
    per_source = {k: v for k, v in per_source.items() if sum(v.values()) > 0}

    if not args.dry_run:
        _write_split_manifest(per_source, output)

    _print_summary(class_dist_by_source, per_source, train_c, val_c, test_c, len(all_pairs), duplicates_excluded, output)

    # Phase 5 — Optional pre-slicing for SAHI train/inference parity
    if args.slice and not args.dry_run:
        _run_slicer(output, args)

    if args.dry_run:
        logger.info("=== DRY RUN COMPLETE — no files were written ===")


def _run_slicer(unified_output: Path, args: argparse.Namespace) -> None:
    """Slice the unified dataset with the same params as SAHI inference.

    Pulls slice parameters from ``config/pipeline.yaml → params.sahi`` so
    training and inference can never drift apart.
    """
    from src.config import settings
    from src.training.slicing import (
        slice_dataset,
        write_sliced_dataset_yaml,
    )

    sahi = settings.params.sahi
    class_names = {v: k for k, v in settings.labels.stage1.model_dump().items()}

    sliced_root = unified_output.parent / f"{unified_output.name}_sliced"
    if sliced_root.exists():
        logger.warning("Sliced output dir already exists: %s (overwriting in place)", sliced_root)

    logger.info("=" * 60)
    logger.info("Phase 5: Pre-slicing for SAHI parity")
    logger.info("  Slice size:    %dx%d", sahi.slice_height, sahi.slice_width)
    logger.info("  Overlap:       %.2f", sahi.overlap_ratio)
    logger.info("  Min area:      %.2f", sahi.min_area_ratio)
    logger.info("  Workers:       %d", args.slice_workers)
    logger.info("  Output:        %s", sliced_root)

    results = slice_dataset(
        source_images_dir=unified_output / "images",
        source_labels_dir=unified_output / "labels",
        output_dir=sliced_root,
        splits=("train", "val", "test"),
        slice_height=sahi.slice_height,
        slice_width=sahi.slice_width,
        overlap_height_ratio=sahi.overlap_ratio,
        overlap_width_ratio=sahi.overlap_ratio,
        min_area_ratio=sahi.min_area_ratio,
        num_workers=args.slice_workers,
    )

    for split, stats in results.items():
        logger.info(
            "  [%s] %d images → %d slices, kept=%d dropped=%d in %.1fs",
            split, stats["images_in"], stats["slices_out"],
            stats["kept_boxes"], stats["dropped_boxes"], stats["elapsed_sec"],
        )

    write_sliced_dataset_yaml(sliced_root, class_names)
    logger.info("Sliced dataset ready: %s", sliced_root)


if __name__ == "__main__":
    main()
