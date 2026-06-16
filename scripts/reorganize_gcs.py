#!/usr/bin/env python3
"""
Reorganise the GCS bucket ``catnip-data`` for Stage 1 training.

Operations (in order):
  1. Archive (or delete) stale training runs
  2. Create Stage 1 directory prefixes
  3. Download the latest Label Studio export

Usage::

    python scripts/reorganize_gcs.py              # archive stale runs, create structure
    python scripts/reorganize_gcs.py --dry-run    # preview without executing
    python scripts/reorganize_gcs.py --delete     # permanently delete instead of archive
    python scripts/reorganize_gcs.py --force      # skip confirmation prompts
"""

import argparse
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("reorganize_gcs")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BUCKET = "gs://catnip-data"
ARCHIVE_BASE = f"{BUCKET}/archive"

# Paths to archive or delete (relative to bucket root).
STALE_PATHS = [
    "runs/izutsumi*",
    "runs/0.11.*",
    "runs/detect/",
    "results/",
]

# Paths that MUST remain untouched (validated before any mutation).
PROTECTED = [
    "data/manga/",
    "data/annotations/",
]

# Stage 1 GCS prefixes to create.
STAGE1_PREFIXES = [
    "data/stage1/images/",
    "data/stage1/labels/",
]

# Where the latest Label Studio export should land.
ANNOTATIONS_EXPORT = "data/annotations/latest_export.json"


# ---------------------------------------------------------------------------
# Guard: gsutil availability (Law 4 — Fail Fast)
# ---------------------------------------------------------------------------

def _gsutil_exists() -> bool:
    """Return True if ``gsutil`` is on PATH."""
    try:
        subprocess.run(
            ["gsutil", "version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ---------------------------------------------------------------------------
# gsutil runner
# ---------------------------------------------------------------------------

def _run(
    cmd: list[str],
    dry_run: bool,
    *,
    stdin_input: str | None = None,
) -> subprocess.CompletedProcess:
    """Execute a gsutil command list, or print it when *dry_run* is True.

    Pass *stdin_input* to pipe content to the subprocess's stdin
    (used for ``gsutil cp -`` style commands).
    """
    cmd_str = " ".join(cmd)
    if dry_run:
        logger.info("  [DRY-RUN] %s", cmd_str)
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    logger.debug("  Executing: %s", cmd_str)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        input=stdin_input,
    )
    if result.returncode != 0:
        logger.error(
            "  gsutil exited %d: %s",
            result.returncode,
            result.stderr.strip(),
        )
    return result


def _ls(path: str) -> list[str]:
    """List objects at *path*, returning one entry per line (may be empty)."""
    result = subprocess.run(
        ["gsutil", "ls", path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.strip().split("\n") if line]


# ---------------------------------------------------------------------------
# Validation (Law 2 — Parse, Don't Validate)
# ---------------------------------------------------------------------------

def _overlaps_protected(path: str) -> bool:
    """Return True if *path* shares a protected prefix."""
    path = path.rstrip("*").rstrip("/") + "/"
    for protected in PROTECTED:
        protected = protected.rstrip("/") + "/"
        if path.startswith(protected) or protected.startswith(path):
            return True
    return False


# ---------------------------------------------------------------------------
# Operation 1: Archive / delete stale paths
# ---------------------------------------------------------------------------

def _archive_stale_paths(*, delete: bool, dry_run: bool) -> dict:
    """Archive (mv → archive/) or delete stale GCS objects.

    Returns a summary dict keyed by GCS path with action and object count.
    """
    summary: dict = {}

    for rel_path in STALE_PATHS:
        full_gs = f"{BUCKET}/{rel_path}"

        if _overlaps_protected(rel_path):
            logger.error(
                "  SAFETY: %s overlaps a protected path. Skipping.",
                full_gs,
            )
            summary[full_gs] = {"action": "skipped (protected)", "count": 0}
            continue

        existing = _ls(full_gs)
        count = len(existing)

        if count == 0:
            logger.info("  %s — no objects found, skipping.", full_gs)
            summary[full_gs] = {"action": "none (empty)", "count": 0}
            continue

        is_directory = rel_path.endswith("/") and "*" not in rel_path

        if delete:
            logger.info("  Deleting %s (%d objects)...", full_gs, count)
            if is_directory:
                _run(["gsutil", "-m", "rm", "-r", full_gs], dry_run=dry_run)
            else:
                _run(["gsutil", "-m", "rm", full_gs], dry_run=dry_run)
            summary[full_gs] = {"action": "deleted", "count": count}
        else:
            dest = f"{ARCHIVE_BASE}/{rel_path.rstrip('/').rstrip('*')}"
            logger.info("  Archiving %s → %s (%d objects)...", full_gs, dest, count)
            if is_directory:
                _run(["gsutil", "-m", "mv", "-r", full_gs, dest + "/"], dry_run=dry_run)
            else:
                _run(["gsutil", "-m", "mv", full_gs, dest + "/"], dry_run=dry_run)
            summary[full_gs] = {"action": "archived", "count": count}

    return summary


# ---------------------------------------------------------------------------
# Operation 2: Create Stage 1 structure
# ---------------------------------------------------------------------------

def _create_stage1_structure(*, dry_run: bool) -> dict:
    """Ensure Stage 1 GCS directory prefixes exist."""
    summary: dict = {}
    for prefix in STAGE1_PREFIXES:
        full_gs = f"{BUCKET}/{prefix}"
        logger.info("  Creating prefix %s ...", full_gs)

        # GCS has no native "mkdir".  Writing an empty placeholder
        # object is the standard way to establish a "folder" prefix
        # so that it shows up in console listings.
        _run(
            ["gsutil", "cp", "-", f"{full_gs}.gcsdir"],
            dry_run=dry_run,
            stdin_input="",
        )

        summary[full_gs] = "created"
    return summary


# ---------------------------------------------------------------------------
# Operation 3: Download latest Label Studio export
# ---------------------------------------------------------------------------

def _find_latest_export_remote() -> str | None:
    """Search GCS for the newest export JSON under annotations/."""
    all_objects = _ls(f"{BUCKET}/data/annotations/**")
    json_objects = [o for o in all_objects if o.endswith(".json")]
    if not json_objects:
        return None

    # Retrieve modification times via `gsutil ls -l` bulk listing
    result = subprocess.run(
        ["gsutil", "ls", "-l", f"{BUCKET}/data/annotations/**"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None

    best_time: datetime | None = None
    best_object: str | None = None

    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("TOTAL:") or "gs://" not in line:
            continue
        # Format: 123456789  2026-01-15T10:30:45Z  gs://bucket/path
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            ts = datetime.fromisoformat(parts[1].rstrip("Z")).replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue
        obj_path = parts[2]
        if obj_path.endswith(".json") and (best_time is None or ts > best_time):
            best_time = ts
            best_object = obj_path

    return best_object


def _find_latest_export_local() -> str | None:
    """Search the local project for the newest Label Studio export JSON."""
    project_root = Path(__file__).resolve().parent.parent
    candidates: list[Path] = []

    for pattern in ("data/annotations/**/*.json", "catnip-data/data/annotations/**/*.json"):
        for p in project_root.glob(pattern):
            if p.is_file():
                candidates.append(p)

    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def _download_latest_export(*, dry_run: bool) -> dict:
    """Copy the newest Label Studio export to the canonical location."""
    summary: dict = {"source": None, "destination": f"{BUCKET}/{ANNOTATIONS_EXPORT}"}

    # Prefer remote, fall back to local
    source = _find_latest_export_remote()

    if source:
        logger.info("  Found remote export: %s", source)
        summary["source"] = source
        _run(
            ["gsutil", "cp", source, f"{BUCKET}/{ANNOTATIONS_EXPORT}"],
            dry_run=dry_run,
        )
        summary["action"] = "copied (remote)"
        return summary

    # Try local fallback
    local_source = _find_latest_export_local()
    if local_source is None:
        logger.warning(
            "  No Label Studio export found (remote or local). "
            "Re-run after exporting from Label Studio."
        )
        summary["action"] = "skipped (not found)"
        return summary

    logger.info("  Found local export: %s", local_source)
    summary["source"] = local_source
    _run(
        ["gsutil", "cp", local_source, f"{BUCKET}/{ANNOTATIONS_EXPORT}"],
        dry_run=dry_run,
    )
    summary["action"] = "copied (local)"
    return summary


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(
    archive_summary: dict,
    stage1_summary: dict,
    export_summary: dict,
    dry_run: bool,
) -> None:
    """Print a human-readable summary of everything that happened."""
    mode = "DRY-RUN" if dry_run else "EXECUTED"
    print(f"\n{'=' * 60}")
    print(f"  GCS Reorganise Summary  [{mode}]")
    print(f"{'=' * 60}")

    print("\n  [1] Stale Paths")
    if archive_summary:
        for path, info in archive_summary.items():
            print(f"      {info['action']:>18}  {path}  ({info['count']} objects)")
    else:
        print("      (none)")

    print("\n  [2] Stage 1 Structure")
    if stage1_summary:
        for prefix, status in stage1_summary.items():
            print(f"      {status:>18}  {prefix}")
    else:
        print("      (none)")

    print("\n  [3] Label Studio Export")
    print(f"      {export_summary.get('action', 'unknown'):>18}  {export_summary.get('destination', '')}")
    if export_summary.get("source"):
        print(f"      {'source':>18}  {export_summary['source']}")

    print(f"\n{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Confirmation
# ---------------------------------------------------------------------------

def _confirm(force: bool, dry_run: bool) -> bool:
    """Ask the user to confirm destructive operations."""
    if dry_run:
        return True
    if force:
        return True
    try:
        response = input("\nProceed with GCS reorganisation? [y/N] ").strip().lower()
        return response in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reorganise the catnip-data GCS bucket for Stage 1 training.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what WOULD happen without executing any gsutil commands.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Permanently delete stale paths instead of archiving them.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt (useful in CI/scripts).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )

    args = parser.parse_args()

    # --- Setup logging ---
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(message)s",
    )

    # --- Guard: gsutil must exist (Law 4 — Fail Fast) ---
    if not _gsutil_exists():
        logger.error(
            "gsutil not found on PATH. "
            "Install the Google Cloud SDK: https://cloud.google.com/sdk/docs/install"
        )
        sys.exit(1)

    logger.info("Bucket: %s", BUCKET)
    logger.info(
        "Mode:   %s",
        "DRY-RUN (no changes will be made)" if args.dry_run else
        ("DELETE (permanent removal)" if args.delete else "ARCHIVE (mv to %s)") % ARCHIVE_BASE,
    )

    # --- Confirmation ---
    if not _confirm(force=args.force, dry_run=args.dry_run):
        logger.info("Aborted by user.")
        sys.exit(0)

    # --- Operation 1: Archive / delete stale paths ---
    logger.info("\n[1/3] Processing stale paths...")
    archive_summary = _archive_stale_paths(
        delete=args.delete,
        dry_run=args.dry_run,
    )

    # --- Operation 2: Create Stage 1 structure ---
    logger.info("\n[2/3] Creating Stage 1 directory structure...")
    stage1_summary = _create_stage1_structure(dry_run=args.dry_run)

    # --- Operation 3: Download latest Label Studio export ---
    logger.info("\n[3/3] Downloading latest Label Studio export...")
    export_summary = _download_latest_export(dry_run=args.dry_run)

    # --- Summary ---
    _print_summary(archive_summary, stage1_summary, export_summary, args.dry_run)


if __name__ == "__main__":
    main()
