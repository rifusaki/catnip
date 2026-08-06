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
generates for fast image access account for ~2.8 GB at imgsz=640 (they
mirror the dataset into a flat-binary form).  They are derived state, not
deliverables, and they are regenerated automatically on the next training
session — so we delete them before upload.  A typical publish is
< 50 MB (the trained ``best.pt`` + ``last.pt`` + the run's plots and CSV).

Prereqs (set by the notebook via Kaggle Secrets):

* ``KAGGLE_USERNAME``, ``KAGGLE_KEY`` env vars.
* ``kaggle`` CLI on $PATH (``pip install kaggle``).
* the working dir must contain ``models/`` and ``runs/detect/stage1/``
  (the standard layout that ``scripts/train/stage1.py`` writes to).

Usage (in a Kaggle notebook)::

    !python /kaggle/working/catnip/scripts/kaggle_publish.py \\
        --output-dir /kaggle/working/catnip-data-local \\
        --dataset-slug catnip-stage1-output \\
        --version-notes "phase-A eval, 24 epochs, batch=64"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

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


def _check_prereqs() -> int:
    """Return non-zero if any required tool/env is missing."""
    failed = 0
    if shutil.which("kaggle") is None:
        logger.error("kaggle CLI not found. `pip install kaggle` first.")
        failed += 1
    if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        logger.error(
            "KAGGLE_USERNAME / KAGGLE_KEY not in env. "
            "Add them as Kaggle Secrets and load with kaggle_secrets.UserSecretsClient."
        )
        failed += 1
    return failed


def _materialize_kaggle_json() -> Path:
    """Write ``~/.kaggle/kaggle.json`` from the KAGGLE_* env vars.

    The Kaggle CLI refuses to read credentials from env vars; it insists
    on the JSON file.  This bridges the two so the notebook only needs
    to set env vars (which it does from Kaggle Secrets).
    """
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "username": os.environ["KAGGLE_USERNAME"],
        "key": os.environ["KAGGLE_KEY"],
    }
    kaggle_json.write_text(json.dumps(payload))
    try:
        kaggle_json.chmod(0o600)
    except OSError:
        # Some filesystems (e.g. Windows mounts in WSL) don't support chmod;
        # kaggle still works with mode 644 in that case.
        logger.debug("Could not chmod 0600 on %s; continuing.", kaggle_json)
    logger.info("Wrote %s (mode 0600).", kaggle_json)
    return kaggle_json


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
    is human-browsable on kaggle.com.  ``--dir-mode zip`` is applied at
    upload time to make the transfer a single archive.
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


def _kaggle_dataset_exists(slug: str) -> bool:
    rc = subprocess.call(
        ["kaggle", "datasets", "status", slug],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return rc == 0


def _upload(staging_dir: Path, slug: str, version_notes: str, dry_run: bool) -> int:
    exists = _kaggle_dataset_exists(slug)
    if exists:
        logger.info("Versioning existing Kaggle dataset '%s'", slug)
        cmd = [
            "kaggle", "datasets", "version",
            "-p", str(staging_dir),
            "-m", version_notes,
            "--dir-mode", "zip",
        ]
    else:
        logger.info("Creating new Kaggle dataset '%s' (private)", slug)
        cmd = [
            "kaggle", "datasets", "create",
            "-p", str(staging_dir),
            "-u",
            "--dir-mode", "zip",
        ]
    logger.info("$ %s", " ".join(cmd))
    if dry_run:
        print(" ".join(cmd))
        return 0
    return subprocess.call(cmd)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish Kaggle session outputs to a private dataset.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("/kaggle/working/catnip-data-local"),
        help="Root that contains models/ and runs/ "
             "(default: /kaggle/working/catnip-data-local).",
    )
    parser.add_argument(
        "--staging-dir", type=Path,
        default=Path("/kaggle/working/catnip-stage1-output-staging"),
        help="Working dir for the upload "
             "(default: /kaggle/working/catnip-stage1-output-staging).",
    )
    parser.add_argument(
        "--dataset-slug", default="catnip-stage1-output",
        help="Target Kaggle dataset slug (default: catnip-stage1-output).",
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
    if _check_prereqs() != 0:
        return 1
    _materialize_kaggle_json()
    _pack_outputs(args.output_dir, args.staging_dir)
    rc = _upload(args.staging_dir, args.dataset_slug, args.version_notes, args.dry_run)
    if rc == 0:
        logger.info(
            "Done. Pull the artifact with: kaggle datasets download %s",
            args.dataset_slug,
        )
    return rc


if __name__ == "__main__":
    sys.exit(main())
