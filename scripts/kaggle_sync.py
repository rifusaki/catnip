#!/usr/bin/env python3
"""
kaggle_sync.py - Publish the Stage 1 sliced training dataset to Kaggle.

Kaggle is an optional training-data location for Stage 1 (Lightning AI is the
primary training host).  The notebook expects the attached dataset to contain
this layout:

    training/stage1_sliced/dataset.yaml
    training/stage1_sliced/images/{train,val,test}/...
    training/stage1_sliced/labels/{train,val,test}/...

This script builds that Kaggle payload shape and uploads it via kagglehub,
creating a new dataset version each run.  The dataset must already exist on
Kaggle (created once via the web UI).

Auth uses the modern KAGGLE_API_TOKEN env var (or ~/.kaggle/access_token).
Get the token from https://www.kaggle.com/settings/api → "Generate New Token".

Normal update from a freshly regenerated local dataset::

    python scripts/kaggle_sync.py \\
        --source catnip-data/training/stage1_sliced \\
        --version-notes "post-slicing refresh"

Check the server-side processing status of an uploaded dataset (exits 0 when
READY, 1 when FAILED, 2 when still processing, 3 when DELETED)::

    python scripts/kaggle_sync.py --check-status --kaggle-slug catnip-stage1-v5

Poll for processing completion (loops until terminal state, prints timestamped
status)::

    python scripts/kaggle_sync.py --check-status --kaggle-slug catnip-stage1-v5 \\
        --poll --poll-interval 60

Use ``--dry-run`` to print the transfer plan without running it.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

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
DEFAULT_DATASET_SUBDIR = Path("training") / "stage1_sliced"
DEFAULT_STAGING_DIR = Path(tempfile.gettempdir()) / "catnip-kaggle-stage1-sliced"
DEFAULT_KAGGLE_SLUG = "catnip-stage1-sliced"
STAGING_MARKER = ".catnip-kaggle-staging"

# macOS / editor droppings that Kaggle's backend validator silently rejects
# (see https://github.com/Kaggle/kaggle-cli/issues/430). Scrubbed from the
# staged tree before upload and also passed as ignore_patterns as a belt +
# suspenders measure.
SCRUB_PATTERNS = ["__pycache__/", ".DS_Store", "__MACOSX/", ".AppleDouble", "._*", "*.lock"]

# Pre-zip payload subdir (sibling of --dataset-subdir). When --pre-zip is active, the
# actual upload root becomes <staging>/<PAYLOAD_DIR_NAME>/ which contains
# {dataset.yaml, archive.zip}. Two files → kagglehub never triggers its >50-files
# auto-zip path; Kaggle unzips archive.zip server-side; YOLO layout is preserved.
PAYLOAD_DIR_NAME = ".kaggle-payload"
ARCHIVE_NAME = "archive.zip"
KAGGLE_URL_BASE = "https://www.kaggle.com/datasets"

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

    if not args.skip_kaggle and not _has_api_token():
        logger.error(
            "Kaggle API token not found.  Set KAGGLE_API_TOKEN env var "
            "or create ~/.kaggle/access_token with the token from "
            "https://www.kaggle.com/settings/api"
        )
        failed += 1

    return failed


def _find_unsafe_names(dataset_root: Path) -> list[str]:
    """Return relative paths whose filename has chars Kaggle silently rejects.

    A name is safe iff every character is alphanumeric or one of ``._-``.
    Anything else (apostrophe, space, ``&``, non-ASCII, ...) causes Kaggle's
    backend validator to fail the entire upload without surfacing an error.
    """
    allowed = frozenset(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    )
    unsafe: list[str] = []
    for path in sorted(dataset_root.rglob("*")):
        if path.is_dir():
            continue
        if any(ch not in allowed for ch in path.name):
            unsafe.append(str(path.relative_to(dataset_root)))
    return unsafe


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

    unsafe = _find_unsafe_names(dataset_root)
    if unsafe:
        logger.error(
            "Found %d filename(s) with characters Kaggle silently rejects "
            "(apostrophes, spaces, &, non-ASCII, ...). These kill the whole "
            "upload with no error surfaced (see kaggle-api #430/#524).",
            len(unsafe),
        )
        for name in unsafe[:10]:
            logger.error("  unsafe: %s", name)
        if len(unsafe) > 10:
            logger.error("  ... and %d more", len(unsafe) - 10)
        logger.error(
            "Run scripts/unify/sanitize_names.py to rename to safe, deterministic "
            "names, then point --source at the sanitized output."
        )
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


def _copy_local_source(source: Path, dataset_root: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"copytree {shlex.quote(str(source))} {shlex.quote(str(dataset_root))}")
        return
    shutil.copytree(source, dataset_root, dirs_exist_ok=True)


def _scrub_staging(dataset_root: Path, dry_run: bool) -> int:
    """Remove macOS / editor cruft that causes Kaggle to silently fail processing.

    Returns the number of files removed. Walks the tree bottom-up so directory
    removals under __MACOSX/ don't trip the walker.
    """
    removed = 0
    if dry_run:
        logger.info("Would scrub patterns %s under %s", SCRUB_PATTERNS, dataset_root)
        return 0

    for path in sorted(dataset_root.rglob("*"), reverse=True):
        name = path.name
        kill = (
            name == ".DS_Store"
            or name == ".AppleDouble"
            or name == "__pycache__"  # mirror the ignore_patterns entry
            or name == "__MACOSX"  # dirs singleton, handle below
            or name.startswith("._")  # dot-underscore AppleDouble siblings
            or name.endswith(".lock")
        )
        if not kill:
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed += 1
            logger.debug("scrubbed %s", path)
        except FileNotFoundError:
            pass
    return removed


def _pre_zip_staging(
    staging_dir: Path,
    dataset_subdir: Path,
    payload_dir: Path,
    dry_run: bool,
) -> Path:
    """Pre-build a single archive.zip from the staged YOLO tree using ``zip -r -X``.

    Strips macOS extended attributes that produce @PaxHeader / __MACOSX / ._appleSibling
    entries inside the zip (Kaggle's backend validator silently rejects these).

    The whole *dataset_subdir* (e.g. ``training/stage1_sliced``) is archived with its
    path prefix preserved, so after Kaggle unzips server-side the dataset root layout
    is unchanged::

        training/stage1_sliced/dataset.yaml
        training/stage1_sliced/images/{train,val,test}/...
        training/stage1_sliced/labels/{train,val,test}/...

    Returns payload_dir. With a single archive.zip at the payload root, kagglehub's
    >50-files auto-zip is bypassed (the auto-zip path uses Python's zipfile.ZipFile
    without -X and so reintroduces macOS xattr noise — we pre-zip cleanly instead).
    """
    if dry_run:
        logger.info("Would pre-zip %s -> %s", dataset_subdir, payload_dir)
        return payload_dir

    payload_dir.mkdir(parents=True, exist_ok=True)

    archive_path = payload_dir / ARCHIVE_NAME
    cmd = ["zip", "-r", "-X", str(archive_path), str(dataset_subdir)]
    logger.info("Pre-zipping %s -> %s (zip -r -X)", dataset_subdir, archive_path)
    rc = subprocess.call(cmd, cwd=str(staging_dir))
    if rc != 0:
        logger.error("zip -r -X failed (exit %d).", rc)
        sys.exit(rc)

    size_mb = archive_path.stat().st_size / (1024 * 1024)
    logger.info(
        "Pre-zip complete. Payload root: %s, archive size: %.1f MB",
        payload_dir, size_mb,
    )
    return payload_dir


@contextlib.contextmanager
def _capture_kaggle_create_responses():
    """Monkey-patch kagglesdk's DatasetApiClient.create_dataset and
    .create_dataset_version to capture the responses that kagglehub v1.0.2's
    handle_mutate_call silently discards.

    kagglehub v1.0.2 _create_dataset calls::

        handle_mutate_call(lambda: api_client.datasets.dataset_api_client.create_dataset(r))

    The lambda's return value (ApiCreateDatasetResponse with .status / .error /
    .invalid_tags) is thrown away by handle_mutate_call. This patch wraps the underlying
    kagglesdk method to stash the response in a dict, surfaces it to the caller, and lets
    handle_mutate_call keep doing whatever it does (including raising on HTTP errors).

    Captured dict shape:
        {"create": ApiCreateDatasetResponse | None,
         "version": ApiCreateDatasetResponse | None,
         "exception": Exception | None}
    """
    captured: dict[str, Any] = {"create": None, "version": None, "exception": None}
    try:
        from kagglesdk.datasets.services.dataset_api_service import DatasetApiClient
    except ImportError as e:
        logger.warning(
            "Could not import kagglesdk.DatasetApiClient (%s); "
            "kagglehub v1.0.2 silent-failure bug is unmitigated this run.",
            e,
        )
        yield captured
        return

    original_create = DatasetApiClient.create_dataset
    original_version = DatasetApiClient.create_dataset_version

    def _wrap(fn, key):
        def wrapper(self, request):
            try:
                response = fn(self, request)
                captured[key] = response
                return response
            except Exception as e:
                captured["exception"] = e
                raise
        return wrapper

    DatasetApiClient.create_dataset = _wrap(original_create, "create")
    DatasetApiClient.create_dataset_version = _wrap(original_version, "version")
    try:
        yield captured
    finally:
        DatasetApiClient.create_dataset = original_create
        DatasetApiClient.create_dataset_version = original_version


def _check_dataset_status(handle: str, poll: bool, interval_s: int) -> int:
    """Query Kaggle for the processing status of the latest version of a dataset.

    Calls kaggle.datasets.dataset_api_client.get_dataset_status which returns a
    DatabundleVersionStatus enum (NOT_YET_PERSISTED, BLOBS_RECEIVED,
    BLOBS_DECOMPRESSED, BLOBS_COPIED_TO_SDS, INDIVIDUAL_BLOBS_COMPRESSED,
    READY, FAILED, DELETED, REPROCESSING).

    Exit codes:
        0 — READY
        1 — FAILED
        2 — non-terminal (still processing)
        3 — DELETED
    """
    from kagglehub.handle import parse_dataset_handle
    from kagglehub.clients import build_kaggle_client
    from kagglesdk.datasets.types.dataset_api_service import (
        ApiGetDatasetStatusRequest,
        ApiGetDatasetRequest,
    )

    h = parse_dataset_handle(handle)

    def _stamp(msg: str) -> None:
        print(f"{datetime.datetime.now().isoformat(timespec='seconds')}  {msg}")

    def _current_version_number(api_client) -> int | None:
        """Return current_version_number via /view, or None if unavailable.

        /view (GetDataset) is the source of truth for whether a dataset has a
        successfully-processed version. Returns None both when the slug doesn't
        exist yet (still uploading) and when it is a zombie (placeholder entry,
        no successful version).
        """
        vr = ApiGetDatasetRequest()
        vr.owner_slug = h.owner
        vr.dataset_slug = h.dataset
        try:
            resp = api_client.datasets.dataset_api_client.get_dataset(vr)
            return resp.current_version_number
        except Exception:
            return None

    while True:
        status_name: str | None = None
        with build_kaggle_client() as api_client:
            try:
                sr = ApiGetDatasetStatusRequest()
                sr.owner_slug = h.owner
                sr.dataset_slug = h.dataset
                resp = api_client.datasets.dataset_api_client.get_dataset_status(sr)
                status_name = resp.status.name
                _stamp(f"status={status_name}")
            except Exception as e:
                # GetDatasetStatus raises 403/404 while the dataset is still
                # uploading, is a zombie, or has left the in-flight window
                # (READY). In all cases, /view's current_version_number is the
                # authoritative signal. Don't treat an exception as terminal —
                # keep polling if requested.
                cvn = _current_version_number(api_client)
                if cvn:
                    _stamp(f"READY (/status unavailable, /view current_version_number={cvn})")
                    return 0
                _stamp(
                    f"not yet accessible (/status {type(e).__name__}, "
                    "no version via /view) — still uploading or processing"
                )

        if status_name is not None:
            if status_name == "READY":
                return 0
            if status_name == "FAILED":
                return 1
            if status_name == "DELETED":
                return 3
            # else: non-terminal (NOT_YET_PERSISTED / BLOBS_* / REPROCESSING)

        if not poll:
            return 2

        time.sleep(interval_s)


def _check_slug_availability(
    handle: str,
    allow_existing: bool,
    force_resume_zombie: bool,
) -> int:
    """Pre-flight check: is the target slug safe to upload to?

    Kaggle's /view (GetDataset) returns 403/404 for ANY slug we can't access,
    including brand-new slugs — it hides existence, so a 403/404 cannot
    distinguish "fresh" from "zombie". We therefore only block on an
    unambiguous 200 response:
      - current_version_number set  -> dataset already exists (block unless --allow-existing)
      - current_version_number None -> zombie placeholder (block unless --force-resume-zombie)

    Returns:
        0 — proceed (fresh slug, or existing allowed via flag)
        2 — existing dataset with a version (refused without --allow-existing)
        1 — zombie (refused without --force-resume-zombie)
    """
    from kagglehub.handle import parse_dataset_handle
    from kagglehub.clients import build_kaggle_client
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest

    h = parse_dataset_handle(handle)

    try:
        with build_kaggle_client() as api_client:
            vr = ApiGetDatasetRequest()
            vr.owner_slug = h.owner
            vr.dataset_slug = h.dataset
            resp = api_client.datasets.dataset_api_client.get_dataset(vr)
            cvn = resp.current_version_number
    except Exception as e:
        msg = str(e).lower()
        if "403" in msg or "forbidden" in msg or "404" in msg or "not found" in msg:
            # Fresh or unknown slug — Kaggle hides existence via 403/404, so
            # there is nothing to guard against here. Proceed and let the
            # upload itself surface any real conflict.
            logger.info("Slug %s not visible via /view (fresh or unknown); proceeding.", handle)
            return 0
        raise

    if cvn:
        if allow_existing:
            logger.warning(
                "Dataset %s already exists with current_version_number=%s; "
                "creating a new version (use --no-allow-existing to refuse).",
                handle, cvn,
            )
            return 0
        logger.error(
            "Dataset %s already exists with current_version_number=%s. "
            "Use --allow-existing to create a new version, or use a different --kaggle-slug.",
            handle, cvn,
        )
        return 2

    # 200 with current_version_number=None — registry placeholder with no successful version
    if force_resume_zombie:
        logger.warning(
            "Dataset %s is a zombie (current_version_number=None); "
            "--force-resume-zombie set, proceeding (likely to fail).",
            handle,
        )
        return 0
    logger.error(
        "Dataset %s exists in zombie state (current_version_number=None). "
        "Use a fresh --kaggle-slug, or --force-resume-zombie to attempt anyway.",
        handle,
    )
    return 1


def _prepare_staging(
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    staging_dir = args.staging_dir.resolve()
    dataset_root = staging_dir / args.dataset_subdir

    _reset_staging_dir(staging_dir, args.dry_run)

    source = (args.source or DEFAULT_LOCAL_SOURCE).resolve()
    logger.info("Staging local source %s -> %s", source, dataset_root)
    _validate_dataset_root(source)
    _copy_local_source(source, dataset_root, args.dry_run)

    if not args.dry_run:
        removed = _scrub_staging(dataset_root, args.dry_run)
        if removed:
            logger.info("Scrubbed %d macOS/editor cruft file(s) from staging.", removed)
        _validate_dataset_root(dataset_root)

    return staging_dir, dataset_root


def _upload_to_kaggle_legacy(
    staging_dir: Path,
    handle: str,
    version_notes: str,
    dry_run: bool,
) -> None:
    """Upload the staged dataset using kagglehub's dataset_upload (legacy mode).

    NOTE: kagglehub v1.0.2 silently discards the ApiCreateDatasetResponse.error field,
    so a backend rejection looks identical to success. Prefer --upload-backend verbose
    (the default) which surfaces the real error. Use this legacy mode only if the verbose
    backend breaks against a newer kagglesdk.
    """
    if dry_run:
        print(f"# kagglehub.dataset_upload(handle='{handle}', "
              f"local_dataset_dir='{staging_dir}', version_notes='{version_notes}')")
        return

    logger.info("Uploading to Kaggle dataset '%s' ...", handle)
    kagglehub.dataset_upload(
        handle=handle,
        local_dataset_dir=str(staging_dir),
        version_notes=version_notes,
        ignore_patterns=[STAGING_MARKER, *SCRUB_PATTERNS],
    )
    logger.info("Upload complete: %s", handle)


def _upload_to_kaggle_verbose(
    upload_dir: Path,
    handle: str,
    version_notes: str,
    dry_run: bool,
) -> None:
    """Upload to Kaggle using kagglehub's GCS upload + a wrapped kagglesdk create_dataset
    call that surfaces ApiCreateDatasetResponse.status / .error / .invalid_tags.

    kagglehub v1.0.2's dataset_upload discards those fields and prints a hardcoded
    "Your dataset has been created. Files are being processed..." message regardless
    of the actual server response. This wrapper intercepts the response and aborts
    with the real error if Kaggle's synchronous validator rejected the upload.
    """
    if dry_run:
        print(
            f"# kagglehub.dataset_upload(handle='{handle}', "
            f"local_dataset_dir='{upload_dir}', version_notes='{version_notes}') "
            f"[verbose: response captured]"
        )
        return

    logger.info("Uploading to Kaggle dataset '%s' (verbose backend) ...", handle)
    with _capture_kaggle_create_responses() as captured:
        kagglehub.dataset_upload(
            handle=handle,
            local_dataset_dir=str(upload_dir),
            version_notes=version_notes,
            ignore_patterns=[STAGING_MARKER, *SCRUB_PATTERNS],
        )

    resp = captured.get("create") or captured.get("version")
    err = captured.get("exception")

    if err is not None:
        logger.error("Kaggle create_dataset() raised: %s", err)
        raise err

    if resp is None:
        # Shouldn't happen unless kagglesdk signature changed. Fall back to legacy message.
        logger.warning(
            "Could not capture ApiCreateDatasetResponse (kagglesdk signature changed?). "
            "Falling back to kagglehub's canned status."
        )
        logger.info("Upload complete: %s", handle)
        return

    if resp.error:
        logger.error("Kaggle rejected the upload:")
        logger.error("  error:        %s", resp.error)
        logger.error("  status:       %s", resp.status or "(empty)")
        if resp.invalid_tags:
            logger.error("  invalid_tags: %s", resp.invalid_tags)
        sys.exit(1)

    logger.info("Kaggle: %s", resp.status or "(no status)")
    logger.info("URL: %s/%s", KAGGLE_URL_BASE, handle)
    logger.info("Upload complete: %s", handle)
    logger.info(
        "Poll for completion with:\n"
        "  pixi run -e kaggle kaggle-sync --check-status --kaggle-slug %s --poll",
        handle.split("/")[-1],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish the Stage 1 sliced dataset to Kaggle.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help=f"Local YOLO dataset root to publish (default: {DEFAULT_LOCAL_SOURCE}).",
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
        help="Stage the dataset, but do not upload to Kaggle.",
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
    parser.add_argument(
        "--check-status",
        action="store_true",
        help=(
            "Don't upload — just query Kaggle for the processing status of the "
            "dataset identified by --kaggle-slug and exit. Combine with --poll "
            "to loop. Exit codes: 0=READY, 1=FAILED, 2=still processing, 3=DELETED."
        ),
    )
    parser.add_argument(
        "--poll",
        action="store_true",
        help=(
            "When used with --check-status, loop until the dataset reaches a "
            "terminal state (READY / FAILED / DELETED) instead of exiting after "
            "one check."
        ),
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Seconds between --poll iterations (default: 60).",
    )
    parser.add_argument(
        "--pre-zip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Pre-build a single archive.zip from the staged YOLO tree with "
            "`zip -r -X` (strips macOS xattrs) and upload the payload root "
            "(dataset.yaml + archive.zip), bypassing kagglehub's >50-files "
            "auto-zip path. Default ON; use --no-pre-zip to disable."
        ),
    )
    parser.add_argument(
        "--upload-backend",
        choices=["verbose", "kagglehub"],
        default="verbose",
        help=(
            "How to call Kaggle: 'verbose' (default, surfaces "
            "ApiCreateDatasetResponse.{status,error,invalid_tags}) or 'kagglehub' "
            "(legacy kagglehub.dataset_upload, silently swallows errors)."
        ),
    )
    parser.add_argument(
        "--skip-pre-check",
        action="store_true",
        help=(
            "Skip the pre-flight slug availability check (not recommended; "
            "guards against burning slugs on zombie datasets)."
        ),
    )
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help=(
            "Proceed even if the slug already exists with a current_version_number — "
            "creates a new version of the existing dataset."
        ),
    )
    parser.add_argument(
        "--force-resume-zombie",
        action="store_true",
        help=(
            "Proceed even if the pre-flight check detects a zombie dataset "
            "(current_version_number=None, or HTTP 403). Likely to fail."
        ),
    )
    parser.add_argument(
        "--reuse-payload",
        action="store_true",
        help=(
            "Skip staging, scrubbing, and the ZIP build entirely; reuse an "
            "existing <staging-dir>/.kaggle-payload/ (dataset.yaml + archive.zip) "
            "from a prior run. Fails if the payload is missing."
        ),
    )

    args = parser.parse_args()
    if args.dataset_subdir.is_absolute():
        parser.error("--dataset-subdir must be relative.")
    return args


def main() -> int:
    args = parse_args()
    _setup_logging(args.verbose)

    # --check-status: pure diagnostic mode, no upload.
    if args.check_status:
        handle = _resolve_handle(
            args.kaggle_slug,
            args.kaggle_owner,
            allow_placeholder=args.dry_run,
        )
        return _check_dataset_status(
            handle,
            poll=args.poll,
            interval_s=args.poll_interval,
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

    if args.reuse_payload:
        # Skip the expensive copy+scrub+zip pipeline; reuse the payload built by a
        # prior run. staging_dir/dataset_root are still resolved for the archive
        # path, but _prepare_staging is intentionally not called.
        staging_dir = args.staging_dir.resolve()
        dataset_root = staging_dir / args.dataset_subdir
        payload_dir = staging_dir / PAYLOAD_DIR_NAME
        if not (payload_dir / ARCHIVE_NAME).exists():
            logger.error(
                "--reuse-payload: no %s found at %s. Run once without "
                "--reuse-payload to build it.",
                ARCHIVE_NAME, payload_dir,
            )
            return 1
        upload_dir = payload_dir
    else:
        staging_dir, dataset_root = _prepare_staging(args)
        upload_dir = staging_dir

    if not args.skip_kaggle:
        assert handle is not None

        if not args.skip_pre_check and not args.dry_run:
            rc = _check_slug_availability(
                handle,
                allow_existing=args.allow_existing,
                force_resume_zombie=args.force_resume_zombie,
            )
            if rc == 2:
                return 2
            if rc == 1 and not args.force_resume_zombie:
                return 1

        if not args.reuse_payload and args.pre_zip and not args.dry_run:
            payload_dir = staging_dir / PAYLOAD_DIR_NAME
            # staging-dir is reset each run, so payload_dir won't carry over; safe to wipe.
            if payload_dir.exists():
                shutil.rmtree(payload_dir)
            upload_dir = _pre_zip_staging(
                staging_dir, args.dataset_subdir, payload_dir, args.dry_run,
            )

        if args.upload_backend == "verbose":
            _upload_to_kaggle_verbose(
                upload_dir=upload_dir,
                handle=handle,
                version_notes=args.version_notes,
                dry_run=args.dry_run,
            )
        else:
            _upload_to_kaggle_legacy(
                staging_dir=upload_dir,
                handle=handle,
                version_notes=args.version_notes,
                dry_run=args.dry_run,
            )

    logger.info("Done. Kaggle dataset payload root: %s", upload_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
