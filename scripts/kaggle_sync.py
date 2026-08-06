#!/usr/bin/env python3
"""
kaggle_sync.py - Publish the Stage 1 sliced training dataset to Kaggle.

Kaggle is the primary training-data location for Stage 1.  The notebook
expects the attached dataset to contain this layout:

    training/stage1_sliced/dataset.yaml
    training/stage1_sliced/images/{train,val,test}/...
    training/stage1_sliced/labels/{train,val,test}/...

This script builds that Kaggle payload shape, writes
``dataset-metadata.json`` for the Kaggle CLI, then creates or versions the
private dataset ``catnip-stage1-sliced``.

First-time import from the current GCS copy::

    python scripts/kaggle_sync.py \\
        --source-uri gs://catnip-data/training/stage1_sliced \\
        --staging-dir /Volumes/rifuSSD/catnip-kaggle-stage1-sliced \\
        --expect-new \\
        --version-notes "initial import from GCS"

Normal update from a freshly regenerated local dataset::

    python scripts/kaggle_sync.py \\
        --source catnip-data/training/stage1_sliced \\
        --version-notes "post-slicing refresh"

Optional archive mirror, if/when canonical storage moves to object storage::

    python scripts/kaggle_sync.py \\
        --source catnip-data/training/stage1_sliced \\
        --archive-uri r2:catnip-canonical/training/stage1_sliced

Use ``--dry-run`` to print the transfer and Kaggle commands without running
them.  By default, this script does not write to GCS or R2.
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_SOURCE = PROJECT_ROOT / "catnip-data" / "training" / "stage1_sliced"
DEFAULT_REMOTE_SOURCE = "gs://catnip-data/training/stage1_sliced"
DEFAULT_DATASET_SUBDIR = Path("training") / "stage1_sliced"
DEFAULT_STAGING_DIR = Path(tempfile.gettempdir()) / "catnip-kaggle-stage1-sliced"
DEFAULT_KAGGLE_SLUG = "catnip-stage1-sliced"
DEFAULT_KAGGLE_TITLE = "Catnip Stage 1 Sliced"
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
    """Run a subprocess, or print it in dry-run mode."""
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


def _read_kaggle_username() -> str | None:
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        return None

    try:
        payload = json.loads(kaggle_json.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s: %s", kaggle_json, exc)
        return None

    username = payload.get("username")
    return str(username) if username else None


def _dataset_ref(
    kaggle_slug: str,
    owner: str | None,
    allow_placeholder: bool = False,
) -> tuple[str, str]:
    """Return (full dataset ref, bare slug)."""
    if "/" in kaggle_slug:
        dataset_owner, bare_slug = kaggle_slug.split("/", 1)
        return f"{dataset_owner}/{bare_slug}", bare_slug

    dataset_owner = owner or _read_kaggle_username()
    if not dataset_owner and allow_placeholder:
        dataset_owner = "KAGGLE_USERNAME"
    if not dataset_owner:
        logger.error(
            "Kaggle owner is unknown. Set KAGGLE_USERNAME, create "
            "~/.kaggle/kaggle.json, or pass --kaggle-owner."
        )
        sys.exit(1)

    return f"{dataset_owner}/{kaggle_slug}", kaggle_slug


def _check_prereqs(args: argparse.Namespace) -> int:
    """Return non-zero if any required tool/credential is missing."""
    if args.dry_run:
        return 0

    failed = 0

    if args.source_uri:
        failed += _check_tool(_tool_for_uri(args.source_uri))

    for archive_uri in args.archive_uri:
        failed += _check_tool(_tool_for_uri(archive_uri))

    if not args.skip_kaggle:
        failed += _check_tool("kaggle")
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        env_creds = bool(os.environ.get("KAGGLE_USERNAME")) and bool(
            os.environ.get("KAGGLE_KEY")
        )
        if not kaggle_json.exists() and not env_creds:
            logger.error(
                "Kaggle credentials not found. Provide either "
                "~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY env vars."
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


def _write_dataset_metadata(
    staging_dir: Path,
    dataset_ref: str,
    title: str,
    license_name: str,
    dry_run: bool,
) -> None:
    metadata_path = staging_dir / "dataset-metadata.json"
    payload = {
        "title": title,
        "id": dataset_ref,
        "licenses": [{"name": license_name}],
    }

    if dry_run:
        print(f"write {metadata_path}:")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    logger.info("Wrote Kaggle metadata: %s", metadata_path)


def _prepare_staging(
    args: argparse.Namespace,
    dataset_ref: str | None,
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

    if dataset_ref is not None:
        _write_dataset_metadata(
            staging_dir,
            dataset_ref=dataset_ref,
            title=args.title,
            license_name=args.license_name,
            dry_run=args.dry_run,
        )
    return staging_dir, dataset_root


def _kaggle_dataset_exists(dataset_ref: str) -> bool:
    """True iff the dataset is already registered and accessible."""
    rc = subprocess.call(
        ["kaggle", "datasets", "status", dataset_ref],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return rc == 0


def _upload_to_kaggle(
    staging_dir: Path,
    dataset_ref: str,
    version_notes: str,
    expect_new: bool,
    dry_run: bool,
) -> None:
    """Create the dataset on first run, or version it on subsequent runs."""
    create_cmd = [
        "kaggle",
        "datasets",
        "create",
        "-p",
        str(staging_dir),
        "-u",
        "--dir-mode",
        "zip",
    ]
    version_cmd = [
        "kaggle",
        "datasets",
        "version",
        "-p",
        str(staging_dir),
        "-m",
        version_notes,
        "--dir-mode",
        "zip",
    ]

    if dry_run:
        print(shlex.join(["kaggle", "datasets", "status", dataset_ref]))
        print("# If the dataset does not exist:")
        print(shlex.join(create_cmd))
        print("# If the dataset already exists:")
        print(shlex.join(version_cmd))
        return

    exists = _kaggle_dataset_exists(dataset_ref)
    if expect_new and exists:
        logger.error(
            "Dataset %s already exists, but --expect-new was set.", dataset_ref
        )
        sys.exit(1)

    if exists:
        logger.info("Versioning existing Kaggle dataset '%s'", dataset_ref)
        cmd = version_cmd
    else:
        logger.info("Creating new private Kaggle dataset '%s'", dataset_ref)
        cmd = create_cmd

    rc = _run(cmd, dry_run=False)
    if rc != 0:
        logger.error("Kaggle upload failed (exit %d).", rc)
        sys.exit(rc)


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
            "Kaggle username for dataset-metadata.json. Defaults to "
            "KAGGLE_USERNAME or ~/.kaggle/kaggle.json."
        ),
    )
    parser.add_argument(
        "--title",
        default=DEFAULT_KAGGLE_TITLE,
        help=f"Kaggle dataset title (default: {DEFAULT_KAGGLE_TITLE}).",
    )
    parser.add_argument(
        "--license-name",
        default="unknown",
        help="Kaggle license name for dataset-metadata.json (default: unknown).",
    )
    parser.add_argument(
        "--expect-new",
        action="store_true",
        help="Fail if the Kaggle dataset already exists.",
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

    dataset_ref = (
        None
        if args.skip_kaggle
        else _dataset_ref(
            args.kaggle_slug,
            args.kaggle_owner,
            allow_placeholder=args.dry_run,
        )[0]
    )
    staging_dir, dataset_root = _prepare_staging(args, dataset_ref)

    if not args.skip_kaggle:
        assert dataset_ref is not None
        _upload_to_kaggle(
            staging_dir=staging_dir,
            dataset_ref=dataset_ref,
            version_notes=args.version_notes,
            expect_new=args.expect_new,
            dry_run=args.dry_run,
        )

    for archive_uri in args.archive_uri:
        _archive_dataset(dataset_root, archive_uri, args.archive_mode, args.dry_run)

    logger.info("Done. Kaggle dataset payload root: %s", staging_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
