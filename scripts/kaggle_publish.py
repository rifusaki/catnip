#!/usr/bin/env python3
"""
kaggle_publish.py — Publish a Kaggle notebook session's training outputs to a
private Kaggle dataset so they survive session end.

Designed to run as the third cell of ``notebooks/catnipKaggle.ipynb``.  It
packages the model weights and Ultralytics run directory into a versioned
private dataset (``catnip-stage1-output`` by default).  Subsequent sessions
add new versions; old versions stay in history under Kaggle's 200 GB
private-storage cap.

Why not just ``/kaggle/working/``?  Kaggle's working dir is wiped at the end
of every session, so any model weights or run curves left there vanish
silently.  Publishing to a private dataset is the only built-in way to
make outputs persistent.

Disk-budget housekeeping: the per-split ``*.cache`` files that Ultralytics
generates under ``runs/detect/stage1/`` are stripped before upload (~2.8 GB).

Prereqs (set by the notebook via Kaggle Secrets):

    KAGGLE_API_TOKEN           # from https://www.kaggle.com/settings/api

Usage (in a Kaggle notebook)::

    !python /kaggle/working/catnip/scripts/kaggle_publish.py \\
        --output-dir /kaggle/working/catnip-data-local \\
        --dataset-slug catnip-stage1-output \\
        --version-notes "session $(date -u +%Y-%m-%dT%H:%MZ)"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import kagglehub

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = Path("/kaggle/working/catnip-data-local")
DEFAULT_STAGING_DIR = Path("/kaggle/working/catnip-stage1-output-staging")
DEFAULT_DATASET_SLUG = "catnip-stage1-output"

logger = logging.getLogger("kaggle_publish")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)-8s %(name)-20s %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )


def _resolve_handle(dataset_slug: str, allow_placeholder: bool = False) -> str:
    """Return the full ``owner/slug`` handle."""
    if "/" in dataset_slug:
        return dataset_slug
    try:
        user = kagglehub.whoami(verbose=False)
        return f"{user['username']}/{dataset_slug}"
    except Exception:
        if allow_placeholder:
            return f"YOUR_USERNAME/{dataset_slug}"
        logger.error(
            "Could not resolve Kaggle username.  Set KAGGLE_API_TOKEN env var "
            "or use the full owner/slug form."
        )
        sys.exit(1)


def _check_prereqs(dry_run: bool = False) -> int:
    """Return non-zero if any required env var is missing."""
    if dry_run:
        return 0
    failed = 0
    if not os.environ.get("KAGGLE_API_TOKEN"):
        access_token_file = Path.home() / ".kaggle" / "access_token"
        if not access_token_file.exists():
            logger.error(
                "KAGGLE_API_TOKEN not in env and ~/.kaggle/access_token not found. "
                "Add KAGGLE_API_TOKEN as a Kaggle Secret and set it as an env var."
            )
            failed += 1
    return failed


def _strip_caches(root: Path) -> int:
    """Delete all ``*.cache`` files under ``root`` to fit the 20 GB disk cap.

    Returns the number of files removed.
    """
    removed = 0
    for cache in root.rglob("*.cache"):
        try:
            size_mb = cache.stat().st_size / 1e6
            cache.unlink()
            logger.info("Removed cache %s (%.1f MB)", cache, size_mb)
            removed += 1
        except OSError as exc:
            logger.warning("Could not remove %s: %s", cache, exc)
    return removed


def _pack_outputs(output_dir: Path, staging_dir: Path) -> None:
    """Copy models/ and runs/ into ``staging_dir`` for upload.

    We keep the directory layout (not a single zip) so the uploaded dataset
    is human-browsable on kaggle.com.
    """
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    models_src = output_dir / "models"
    if models_src.exists():
        shutil.copytree(models_src, staging_dir / "models")
        logger.info("Packed models/ → %s/models", staging_dir)
    else:
        logger.warning(
            "No models/ under %s — model may have failed to save.",
            output_dir,
        )

    runs_src = output_dir / "runs" / "detect" / "stage1"
    if runs_src.exists():
        shutil.copytree(runs_src, staging_dir / "runs")
        logger.info("Packed runs/ → %s/runs", staging_dir)
    else:
        logger.warning("No runs/detect/stage1/ under %s.", output_dir)

    _strip_caches(staging_dir)


def _upload(staging_dir: Path, handle: str, version_notes: str, dry_run: bool) -> int:
    logger.info("Uploading to Kaggle dataset '%s' ...", handle)
    if dry_run:
        print(f"# kagglehub.dataset_upload(handle='{handle}', "
              f"local_dataset_dir='{staging_dir}', version_notes='{version_notes}')")
        return 0
    kagglehub.dataset_upload(
        handle=handle,
        local_dataset_dir=str(staging_dir),
        version_notes=version_notes,
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish Kaggle session outputs to a private dataset.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root that contains models/ and runs/ "
             "(default: /kaggle/working/catnip-data-local).",
    )
    parser.add_argument(
        "--staging-dir", type=Path,
        default=DEFAULT_STAGING_DIR,
        help="Working dir for the upload "
             "(default: /kaggle/working/catnip-stage1-output-staging).",
    )
    parser.add_argument(
        "--dataset-slug", default=DEFAULT_DATASET_SLUG,
        help="Target Kaggle dataset slug (default: catnip-stage1-output). "
             "Use owner/slug to avoid resolving username.",
    )
    parser.add_argument(
        "--version-notes", default="auto-publish from notebook session",
        help="Notes attached to the new Kaggle dataset version.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the commands that would run, but do not execute them.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _setup_logging(args.verbose)
    if _check_prereqs(dry_run=args.dry_run) != 0:
        return 1
    handle = _resolve_handle(args.dataset_slug, allow_placeholder=args.dry_run)
    if args.dry_run:
        logger.info("Would pack outputs from %s → %s", args.output_dir, args.staging_dir)
    else:
        _pack_outputs(args.output_dir, args.staging_dir)
    rc = _upload(args.staging_dir, handle, args.version_notes, args.dry_run)
    if rc == 0:
        logger.info(
            "Done. Download the artifact with: "
            "kagglehub.dataset_download('%s')",
            handle,
        )
    return rc


if __name__ == "__main__":
    sys.exit(main())
