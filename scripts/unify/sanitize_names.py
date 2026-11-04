#!/usr/bin/env python3
"""
sanitize_names.py — Rename a YOLO dataset's images/labels to safe, constant,
deterministic names and emit a provenance manifest.

Kaggle's backend silently rejects any dataset containing a filename with
characters it does not allow (apostrophes, ampersands, spaces, non-ASCII, ...).
Manga109 titles such as ``UchiNoNyan'sDiary`` leak apostrophes into filenames
and kill the whole multi-GB upload with no error surfaced. This script renames
every image/label pair to a collision-free, deterministic name derived from the
ORIGINAL stem::

    new_stem = sha256(original_stem.encode("utf-8")).hexdigest()[:16]

The mapping is recorded in a ``manifest.json`` (new stem → provenance) so
credits / attribution can be attached to each element later without ever
re-parsing the original filename.

Properties:
  * Deterministic — same original stem → same new name, on every machine, forever.
  * Constant — adding new images never renames existing ones (new stem → new hash).
  * Safe — hex-only names, valid on Kaggle, S3, GCS, and every filesystem.
  * Unique — SHA-256 truncated to 64 bits; collisions are detected and raised.

Usage::

    python scripts/unify/sanitize_names.py \\
        --input-dir /Volumes/rifuSSD/catnip-data/training/stage1_sliced \\
        --output-dir /Volumes/rifuSSD/catnip-data/training/stage1_sliced_safe

    python scripts/unify/sanitize_names.py \\
        --input-dir catnip-data/training/stage1_sliced --dry-run

The output directory mirrors the input layout (``images/{split}``,
``labels/{split}``) with sanitized stems, plus ``manifest.json`` and a verbatim
copy of ``dataset.yaml`` (and any ``*.json`` metadata at the root).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from pathlib import Path

# Allow running the script directly from scripts/unify/ without an installed package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import settings

logger = logging.getLogger("sanitize_names")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Hex chars kept from the SHA-256 digest. 16 chars = 64 bits of uniqueness.
# Collision probability at 1M files is ~2.7e-8; a collision raises loudly.
HASH_LENGTH = 16

# Filenames are rejected unless every character is in this set.
# (Same rule Kaggle's backend validator effectively applies; see kaggle-api
# #430 / #524 for the silent-failure behavior.)
SAFE_CHARSET = frozenset("abcdefghijklmnopqrstuvwxyz"
                         "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                         "0123456789"
                         "._-")

IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})
LABEL_SUFFIX = ".txt"

MANIFEST_NAME = "manifest.json"

DEFAULT_OUTPUT_SUFFIX = "_safe"


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

def safe_stem(original_stem: str) -> str:
    """Return the deterministic safe stem for *original_stem*.

    SHA-256 is content-independent (hashes the name, not the bytes), so it is
    cheap, reproducible, and constant under incremental dataset growth.
    """
    digest = hashlib.sha256(original_stem.encode("utf-8")).hexdigest()
    return digest[:HASH_LENGTH]


def is_safe_name(name: str) -> bool:
    """True if *name* contains only characters Kaggle accepts."""
    return all(ch in SAFE_CHARSET for ch in name)


def is_excluded(
    stem: str,
    exclude_sources: list[str],
    exclude_titles: list[str],
) -> bool:
    """True if *stem* matches an --exclude-source or --exclude-title pattern.

    Sources are matched against the leading ``_``-delimited token (e.g.
    ``manga109``); titles are matched as case-sensitive substrings anywhere in
    the stem (e.g. ``LoveHina``), which survives the varying
    ``{title}_{volume}_{page}_{bbox}`` layout.
    """
    if stem.split("_", 1)[0] in exclude_sources:
        return True
    return any(t in stem for t in exclude_titles)


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def collect_pairs(input_dir: Path) -> tuple[list[tuple[Path, Path, str]], int]:
    """Collect (image_path, label_path, split_name) triples.

    Walks ``images/<split>/`` generically (train/val/test or any other split
    name) and mirrors each image to ``labels/<split>/<stem>.txt``. Images whose
    label is missing (or labels whose image is missing) are skipped with a
    warning so we never emit orphaned files.
    """
    images_dir = input_dir / "images"
    labels_dir = input_dir / "labels"

    if not images_dir.is_dir():
        logger.error("images/ not found under %s", input_dir)
        sys.exit(1)
    if not labels_dir.is_dir():
        logger.error("labels/ not found under %s", input_dir)
        sys.exit(1)

    pairs: list[tuple[Path, Path, str]] = []
    skipped = 0

    for split_dir in sorted(images_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        split_name = split_dir.name
        labels_split = labels_dir / split_name
        if not labels_split.is_dir():
            logger.warning("labels/%s/ missing — skipping split.", split_name)
            continue

        for image in sorted(split_dir.iterdir()):
            if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label = labels_split / f"{image.stem}{LABEL_SUFFIX}"
            if not label.is_file():
                logger.warning("No label for %s — skipping.", image.name)
                skipped += 1
                continue
            pairs.append((image, label, split_name))

    # Also flag labels with no matching image (so nothing silently vanishes).
    for split_dir in sorted(labels_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        images_split = images_dir / split_dir.name
        for label in sorted(split_dir.glob(f"*{LABEL_SUFFIX}")):
            if not any((images_split / f"{label.stem}{s}").is_file()
                       for s in IMAGE_SUFFIXES):
                logger.warning("No image for %s — label orphaned.", label.name)

    return pairs, skipped


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _prepare_output(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def copy_and_build_manifest(
    pairs: list[tuple[Path, Path, str]],
    input_dir: Path,
    output_dir: Path,
) -> dict:
    """Copy renamed files and return the manifest entries dict.

    Raises if a hash collision is detected (should be astronomically unlikely,
    but Law 4 — fail fast rather than silently overwrite).
    """
    entries: dict[str, dict] = {}
    for image, label, split_name in pairs:
        original_stem = image.stem
        new_stem = safe_stem(original_stem)

        if new_stem in entries:
            raise RuntimeError(
                f"SHA-256({HASH_LENGTH * 4}-bit) collision between "
                f"'{entries[new_stem]['original']}' and '{original_stem}'. "
                "Bump HASH_LENGTH and re-run."
            )

        images_out = output_dir / "images" / split_name
        labels_out = output_dir / "labels" / split_name
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        new_image = images_out / f"{new_stem}{image.suffix.lower()}"
        new_label = labels_out / f"{new_stem}{LABEL_SUFFIX}"

        shutil.copy2(image, new_image)
        shutil.copy2(label, new_label)

        entries[new_stem] = {
            "original": original_stem,
            "source": original_stem.split("_", 1)[0],
            "image": str(image.relative_to(input_dir)),
            "label": str(label.relative_to(input_dir)),
        }

    return entries


def _copy_metadata(input_dir: Path, output_dir: Path) -> None:
    """Copy dataset.yaml and any root-level *.json metadata verbatim."""
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and (path.name == "dataset.yaml" or path.suffix == ".json"):
            shutil.copy2(path, output_dir / path.name)
            logger.debug("Copied metadata: %s", path.name)


# ---------------------------------------------------------------------------
# Summary (Law 5 — Intentional Naming)
# ---------------------------------------------------------------------------

def _print_summary(n_pairs: int, n_skipped: int, n_unsafe: int, output_dir: Path) -> None:
    print(f"\n{'=' * 60}\n{' Filename Sanitization — Complete '.center(60, '=')}\n{'=' * 60}")
    print(f"  Images renamed:     {n_pairs:>6}")
    print(f"  Unsafe before:      {n_unsafe:>6}  (renamed to safe stems)")
    print(f"  Skipped (no pair):  {n_skipped:>6}")
    print(f"  Manifest:           {output_dir / MANIFEST_NAME}")
    print(f"  Output:             {output_dir.resolve()}")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rename YOLO dataset files to safe, deterministic names.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=settings.paths.stage1_sliced_dir,
        help=(
            "YOLO dataset root to sanitize "
            f"(default from config: {settings.paths.stage1_sliced_dir})."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory (default: <input-dir>"
            f"{DEFAULT_OUTPUT_SUFFIX})."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the rename plan and stats without writing anything.",
    )
    parser.add_argument(
        "--exclude-source",
        action="append",
        default=[],
        metavar="SRC",
        help=(
            "Exclude an entire source prefix (repeatable), e.g. --exclude-source "
            "manga109. Useful for dropping whole corpora that trip content review."
        ),
    )
    parser.add_argument(
        "--exclude-title",
        action="append",
        default=[],
        metavar="TITLE",
        help=(
            "Exclude a manga title by substring match (repeatable), e.g. "
            "--exclude-title LoveHina. Matches the source name embedded in the "
            "filename, not the sanitized hex stem."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = Path(str(args.input_dir) + DEFAULT_OUTPUT_SUFFIX)

    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(message)s",
    )

    input_dir = args.input_dir
    if not input_dir.is_dir():
        logger.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    logger.info("Input:  %s", input_dir)
    logger.info("Output: %s%s", args.output_dir, "  (dry run)" if args.dry_run else "")

    pairs, skipped = collect_pairs(input_dir)
    if not pairs:
        logger.error("No image/label pairs found under %s", input_dir)
        sys.exit(1)

    n_excluded = 0
    if args.exclude_source or args.exclude_title:
        before = len(pairs)
        pairs = [
            p for p in pairs
            if not is_excluded(p[0].stem, args.exclude_source, args.exclude_title)
        ]
        n_excluded = before - len(pairs)
        logger.info(
            "Excluded %d pair(s) by --exclude-source/--exclude-title.", n_excluded,
        )
        if not pairs:
            logger.error("All pairs excluded — nothing left to sanitize.")
            sys.exit(1)

    n_unsafe = sum(
        1 for image, _, _ in pairs if not is_safe_name(image.name)
    )

    if args.dry_run:
        logger.info("=== DRY RUN — no files will be written ===")
        logger.info(
            "  pairs=%d, skipped=%d, excluded=%d, unsafe=%d",
            len(pairs), skipped, n_excluded, n_unsafe,
        )
        for image, _, split_name in pairs[:10]:
            logger.info("  [%s] %s -> %s%s", split_name, image.name, safe_stem(image.stem), image.suffix.lower())
        if len(pairs) > 10:
            logger.info("  ... and %d more", len(pairs) - 10)
        logger.info("=== DRY RUN COMPLETE — no files were written ===")
        return

    _prepare_output(args.output_dir)
    entries = copy_and_build_manifest(pairs, input_dir, args.output_dir)
    _copy_metadata(input_dir, args.output_dir)

    manifest = {
        "scheme": f"sha256(original_stem)[:{HASH_LENGTH}]",
        "description": (
            "Deterministic safe-name mapping for portable/Kaggle uploads. "
            "Keys are the safe stem (no extension); 'original' is the pre-rename "
            "stem; 'source' is the best-effort source prefix; 'image'/'label' are "
            "the original relative paths. Attach credits/provenance per entry."
        ),
        "entries": entries,
    }
    manifest_path = args.output_dir / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    _print_summary(len(pairs), skipped, n_unsafe, args.output_dir)


if __name__ == "__main__":
    main()
