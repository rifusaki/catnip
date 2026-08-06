#!/usr/bin/env python3
"""
kaggle_sync.py - Publish the Stage 1 sliced training dataset to Kaggle.

Kaggle is the primary training-data location for Stage 1.  The notebook
expects the attached dataset to contain this layout:

    training/stage1_sliced/dataset.yaml
    training/stage1_sliced/images/{train,val,test}/...
    training/stage1_sliced/labels/{train,val,test}/...

This script builds that Kaggle payload shape and uploads it via kagglehub,
creating a new dataset version each run.  The dataset must already exist on
Kaggle (created once via the web UI).

Auth uses the modern KAGGLE_API_TOKEN env var (or ~/.kaggle/access_token).
Get the token from https://www.kaggle.com/settings/api → "Generate New Token".

First-time import from the current GCS copy::

    python scripts/kaggle_sync.py \\
        --source-uri gs://catnip-data/training/stage1_sliced \\
        --staging-dir /Volumes/rifuSSD/catnip-kaggle-stage1-sliced \\
        --version-notes "initial import from GCS"

Normal update from a freshly regenerated local dataset::

    python scripts/kaggle_sync.py \\
        --source catnip-data/training/stage1_sliced \\
        --version-notes "post-slicing refresh"

Optional archive mirror::

    python scripts/kaggle_sync.py \\
        --source catnip-data/training/stage1_sliced \\
        --archive-uri r2:catnip-canonical/training/stage1_sliced

Use ``--dry-run`` to print the transfer plan without running it.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import kagglehub  # noqa: E402 (after load_dotenv to pick up KAGGLE_API_TOKEN from .env)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def _catnip_data_root() -> Path:
    """Resolve the catnip-data root from ``CATNIP_DATA`` env var or default."""
    env = os.environ.get("CATNIP_DATA")
    if env:
        return Path(env)
    return PROJECT_ROOT / "catnip-data"

DEFAULT_LOCAL_SOURCE = _catnip_data_root() / "training" / "stage1_sliced"
DEFAULT_REMOTE_SOURCE = "gs://catnip-data/training/stage1_sliced"
DEFAULT_DATASET_SUBDIR = Path("training") / "stage1_sliced"
DEFAULT_STAGING_DIR = Path(tempfile.gettempdir()) / "catnip-kaggle-stage1-sliced"
DEFAULT_KAGGLE_SLUG = "catnip-stage1-sliced"
STAGING_MARKER = ".catnip-kaggle-staging"

logger = logging.getLogger("kaggle_sync")


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


def _run(cmd: list[str], dry_run: bool) -> int:
    logger.debug("$ %s", shlex.join(cmd))
    if dry_run:
        print(shlex.join(cmd))
        return 0
    return subprocess.call(cmd)


def _is_gs_uri(uri: str) -> bool:
    return uri.startswith("gs://")


def _uses_rclone(uri: str) -> bool:
    return not _is_gs_uri(uri) and ":" in uri


def _tool_for_uri(uri: str) -> str:
    if _is_gs_uri(uri):
        return "gcloud"
    if _uses_rclone(uri):
        return "rclone"
    raise ValueError(
        f"Unsupported URI '{uri}'. Use gs://... or an rclone remote like "
        "r2:bucket/path."
    )


def _check_tool(tool: str) -> int:
    if shutil.which(tool) is None:
        logger.error("%s not found on $PATH.", tool)
        return 1
    return 0


def _resolve_handle(
    kaggle_slug: str,
    owner: str | None,
    allow_placeholder: bool = False,
) -> str:
    """Return the full ``owner/slug`` dataset handle."""
    if "/" in kaggle_slug:
        return kaggle_slug

    if owner:
        return f"{owner}/{kaggle_slug}"

    try:
        user = kagglehub.whoami(verbose=False)
        return f"{user['username']}/{kaggle_slug}"
    except Exception:
        if allow_placeholder:
            return f"YOUR_USERNAME/{kaggle_slug}"
        logger.error(
            "Could not resolve Kaggle username.  Either pass --kaggle-owner, "
            "set KAGGLE_API_TOKEN, or create ~/.kaggle/access_token."
        )
        sys.exit(1)


def _has_api_token() -> bool:
    """Check whether a Kaggle API token is available."""
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    access_token_file = Path.home() / ".kaggle" / "access_token"
    if access_token_file.exists():
        return True
    return False


def _check_prereqs(args: argparse.Namespace) -> int:
    """Return non-zero if any required tool/credential is missing."""
    if args.dry_run:
        return 0

    failed = 0

    if args.source_uri:
        failed += _check_tool(_tool_for_uri(args.source_uri))

    for archive_uri in args.archive_uri:
        failed += _check_tool(_tool_for_uri(archive_uri))

    if not args.skip_kaggle and not _has_api_token():
        logger.error(
            "Kaggle API token not found.  Set KAGGLE_API_TOKEN env var "
            "or create ~/.kaggle/access_token with the token from "
            "https://www.kaggle.com/settings/api"
        )
        failed += 1

    return failed


def _validate_dataset_root(dataset_root: Path) -> None:
    """Refuse to upload anything that does not look like a YOLO dataset root."""
    if not dataset_root.exists():
        logger.error("Dataset root does not exist: %s", dataset_root)
        sys.exit(1)
    if not dataset_root.is_dir():
        logger.error("Dataset root is not a directory: %s", dataset_root)
        sys.exit(1)

    dataset_yaml = dataset_root / "dataset.yaml"
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"

    if not dataset_yaml.exists():
        logger.error("Missing dataset.yaml at %s.", dataset_yaml)
        sys.exit(1)
    if not images_dir.is_dir():
        logger.error("Missing images/ directory at %s.", images_dir)
        sys.exit(1)
    if not labels_dir.is_dir():
        logger.error("Missing labels/ directory at %s.", labels_dir)
        sys.exit(1)
    if not any(images_dir.rglob("*")):
        logger.error("images/ is empty under %s.", dataset_root)
        sys.exit(1)
    if not any(labels_dir.rglob("*")):
        logger.error("labels/ is empty under %s.", dataset_root)
        sys.exit(1)


def _reset_staging_dir(staging_dir: Path, dry_run: bool) -> None:
    """Create an empty staging dir, refusing to delete unmarked directories."""
    marker = staging_dir / STAGING_MARKER

    if dry_run:
        logger.info("Would reset staging dir: %s", staging_dir)
        return

    if staging_dir.exists():
        if not marker.exists():
            logger.error(
                "Refusing to remove unmarked staging dir: %s. Use a fresh "
                "--staging-dir or delete it manually if it is safe.",
                staging_dir,
            )
            sys.exit(1)
        shutil.rmtree(staging_dir)

    staging_dir.mkdir(parents=True)
    marker.write_text("catnip kaggle staging directory\n")


def _transfer_cmd(source: str, destination: str, mode: str) -> list[str]:
    """Build a remote transfer command for gcloud storage or rclone."""
    remote_side = (
        source if (_is_gs_uri(source) or _uses_rclone(source)) else destination
    )
    tool = _tool_for_uri(remote_side)

    if tool == "gcloud":
        cmd = ["gcloud", "storage", "rsync", "--recursive", source, destination]
        if mode == "sync":
            cmd.insert(3, "--delete-unmatched-destination-objects")
        return cmd

    rclone_action = "sync" if mode == "sync" else "copy"
    return [
        "rclone",
        rclone_action,
        source,
        destination,
        "--progress",
        "--transfers",
        "16",
        "--checkers",
        "32",
    ]


def _copy_local_source(source: Path, dataset_root: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"copytree {shlex.quote(str(source))} {shlex.quote(str(dataset_root))}")
        return
    shutil.copytree(source, dataset_root, dirs_exist_ok=True)


def _prepare_staging(
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    staging_dir = args.staging_dir.resolve()
    dataset_root = staging_dir / args.dataset_subdir

    _reset_staging_dir(staging_dir, args.dry_run)

    if args.source_uri:
        logger.info(
            "Materializing remote source %s -> %s",
            args.source_uri,
            dataset_root,
        )
        if not args.dry_run:
            dataset_root.parent.mkdir(parents=True, exist_ok=True)
        rc = _run(
            _transfer_cmd(args.source_uri, str(dataset_root), "copy"),
            args.dry_run,
        )
        if rc != 0:
            logger.error("Remote source materialization failed (exit %d).", rc)
            sys.exit(rc)
    else:
        source = args.source or DEFAULT_LOCAL_SOURCE
        source = source.resolve()
        logger.info("Staging local source %s -> %s", source, dataset_root)
        _validate_dataset_root(source)
        _copy_local_source(source, dataset_root, args.dry_run)

    if not args.dry_run:
        _validate_dataset_root(dataset_root)

    return staging_dir, dataset_root


def _upload_to_kaggle(
    staging_dir: Path,
    handle: str,
    version_notes: str,
    dry_run: bool,
) -> None:
    """Upload the staged dataset to Kaggle using kagglehub."""
    if dry_run:
        print(f"# kagglehub.dataset_upload(handle='{handle}', "
              f"local_dataset_dir='{staging_dir}', version_notes='{version_notes}')")
        return

    logger.info("Uploading to Kaggle dataset '%s' ...", handle)
    kagglehub.dataset_upload(
        handle=handle,
        local_dataset_dir=str(staging_dir),
        version_notes=version_notes,
        ignore_patterns=[STAGING_MARKER, "__pycache__/"],
    )
    logger.info("Upload complete: %s", handle)


def _archive_dataset(
    dataset_root: Path,
    archive_uri: str,
    archive_mode: str,
    dry_run: bool,
) -> None:
    logger.info("Archiving %s -> %s [%s]", dataset_root, archive_uri, archive_mode)
    rc = _run(
        _transfer_cmd(str(dataset_root), archive_uri, archive_mode),
        dry_run=dry_run,
    )
    if rc != 0:
        logger.error("Archive transfer failed for %s (exit %d).", archive_uri, rc)
        sys.exit(rc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish the Stage 1 sliced dataset to Kaggle.",
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--source",
        type=Path,
        default=None,
        help=(
            "Local YOLO dataset root to publish. Defaults to "
            f"{DEFAULT_LOCAL_SOURCE} when --source-uri is omitted."
        ),
    )
    source_group.add_argument(
        "--source-uri",
        default=None,
        help=(
            "Remote YOLO dataset root to materialize before upload, e.g. "
            f"{DEFAULT_REMOTE_SOURCE} or gcs:catnip-data/training/stage1_sliced."
        ),
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=DEFAULT_STAGING_DIR,
        help=(
            "Local staging root for the Kaggle payload "
            f"(default: {DEFAULT_STAGING_DIR})."
        ),
    )
    parser.add_argument(
        "--dataset-subdir",
        type=Path,
        default=DEFAULT_DATASET_SUBDIR,
        help=(
            "Path inside the Kaggle dataset where the YOLO root is placed "
            f"(default: {DEFAULT_DATASET_SUBDIR})."
        ),
    )
    parser.add_argument(
        "--kaggle-slug",
        default=DEFAULT_KAGGLE_SLUG,
        help=(
            "Kaggle dataset slug or owner/slug "
            f"(default: {DEFAULT_KAGGLE_SLUG})."
        ),
    )
    parser.add_argument(
        "--kaggle-owner",
        default=None,
        help=(
            "Kaggle username for the dataset handle. "
            "Defaults to the authenticated user from kagglehub.whoami()."
        ),
    )
    parser.add_argument(
        "--skip-kaggle",
        action="store_true",
        help="Stage and optionally archive the dataset, but do not upload to Kaggle.",
    )
    parser.add_argument(
        "--archive-uri",
        action="append",
        default=[],
        help=(
            "Optional archive mirror destination. May be repeated. Supports "
            "gs://... and rclone remotes such as r2:bucket/path."
        ),
    )
    parser.add_argument(
        "--archive-mode",
        choices=["copy", "sync"],
        default="copy",
        help=(
            "Archive transfer mode. copy is non-destructive; sync deletes "
            "destination extras (default: copy)."
        ),
    )
    parser.add_argument(
        "--skip-gcs",
        action="store_true",
        help="Deprecated no-op. GCS archive writes are skipped unless --archive-uri is set.",
    )
    parser.add_argument(
        "--version-notes",
        type=str,
        default="manual sync",
        help="Notes attached to a new Kaggle dataset version.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and metadata without transferring or uploading.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    args = parser.parse_args()
    if args.dataset_subdir.is_absolute():
        parser.error("--dataset-subdir must be relative.")
    return args


def main() -> int:
    args = parse_args()
    _setup_logging(args.verbose)

    if args.skip_gcs:
        logger.warning(
            "--skip-gcs is deprecated; GCS is skipped unless --archive-uri is set."
        )

    if _check_prereqs(args) != 0:
        return 1

    handle = (
        None
        if args.skip_kaggle
        else _resolve_handle(
            args.kaggle_slug,
            args.kaggle_owner,
            allow_placeholder=args.dry_run,
        )
    )
    staging_dir, dataset_root = _prepare_staging(args)

    if not args.skip_kaggle:
        assert handle is not None
        _upload_to_kaggle(
            staging_dir=staging_dir,
            handle=handle,
            version_notes=args.version_notes,
            dry_run=args.dry_run,
        )

    for archive_uri in args.archive_uri:
        _archive_dataset(dataset_root, archive_uri, args.archive_mode, args.dry_run)

    logger.info("Done. Kaggle dataset payload root: %s", staging_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
