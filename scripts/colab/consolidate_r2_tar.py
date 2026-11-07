#!/usr/bin/env python3
"""Consolidate multi-part dataset archive on Cloudflare R2 into a single unified .tar.

Replaces split .part_* chunks (previously required due to wrangler's 300 MiB cap)
with a single canonical .tar object using S3 multipart upload.

Workflow:
1. Validate R2 credentials and free local disk space (~15 GB for the unified tar).
2. Scan all split parts for the given prefix and verify part ordering.
3. Stream-download and concatenate parts sequentially into a single local .tar.
4. Verify archive byte size and tar structure.
5. Upload the consolidated .tar to R2 via S3 multipart upload.
6. Verify remote object via s3.head_object (ContentLength match).
7. Safely delete the split .part_* chunks from R2 only after verification passes.
"""

import argparse
import logging
import os
import shutil
import sys
import tarfile
from pathlib import Path

try:
    import boto3
    from boto3.s3.transfer import TransferConfig
except ImportError:
    boto3 = None
    TransferConfig = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

logger = logging.getLogger("catnip.consolidate")

DEFAULT_ENDPOINT = "https://eccb0b608bd926e793224944aba32cf4.r2.cloudflarestorage.com"
DEFAULT_BUCKET = "catnip-canonical"
DEFAULT_KEY = "compressed/stage1_v2_sliced.tar"


def get_s3_client(
    endpoint: str = DEFAULT_ENDPOINT,
    access_key: str | None = None,
    secret_key: str | None = None,
):
    """Create and return a boto3 S3 client configured for Cloudflare R2."""
    if boto3 is None:
        raise RuntimeError("boto3 is required. Run 'pip install boto3'")

    ak = access_key or os.environ.get("R2_ACCESS_KEY_ID")
    sk = secret_key or os.environ.get("R2_SECRET_ACCESS_KEY")
    if not ak or not sk:
        raise ValueError(
            "Missing R2 credentials. Set R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY, "
            "or configure them in Colab secrets."
        )

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=ak,
        aws_secret_access_key=sk,
        region_name="auto",
    )


def list_split_parts(s3_client, bucket: str, unified_key: str) -> list[dict]:
    """Scan and return lexicographically sorted split part objects from R2."""
    prefix = unified_key + ".part_"
    paginator = s3_client.get_paginator("list_objects_v2")
    parts = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            parts.append(item)

    parts.sort(key=lambda p: p["Key"])
    return parts


def stream_download_parts(
    s3_client,
    bucket: str,
    parts: list[dict],
    output_path: Path,
) -> int:
    """Download and stream-concatenate parts into output_path. Returns total bytes written."""
    total_expected_bytes = sum(p["Size"] for p in parts)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()

    bytes_written = 0
    pbar = None
    if tqdm is not None:
        pbar = tqdm(
            total=total_expected_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Downloading & joining",
        )

    with open(output_path, "wb") as out_f:
        for index, part_info in enumerate(parts):
            key = part_info["Key"]
            logger.info("Streaming part %d/%d: %s (%.1f MB)", index + 1, len(parts), key, part_info["Size"] / 1e6)
            resp = s3_client.get_object(Bucket=bucket, Key=key)
            body = resp["Body"]
            while True:
                chunk = body.read(16 * 1024 * 1024)  # 16 MB chunks
                if not chunk:
                    break
                out_f.write(chunk)
                bytes_written += len(chunk)
                if pbar:
                    pbar.update(len(chunk))

    if pbar:
        pbar.close()

    if bytes_written != total_expected_bytes:
        if output_path.exists():
            output_path.unlink()
        raise IOError(
            f"Download byte count mismatch: wrote {bytes_written} bytes, "
            f"expected {total_expected_bytes} bytes."
        )

    return bytes_written


def verify_tar_integrity(tar_path: Path, expected_size: int) -> None:
    """Validate archive size and inspect tar headers."""
    actual_size = tar_path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"Integrity check failed: local size {actual_size} != expected {expected_size}"
        )

    logger.info("Verifying tar structure for %s...", tar_path)
    with tarfile.open(tar_path, "r") as tar:
        first_members = []
        for i, member in enumerate(tar):
            first_members.append(member.name)
            if i >= 10:
                break
    if not first_members:
        raise ValueError("Archive is empty or unreadable as a tar file.")
    logger.info("Tar structure verified. Sample entries: %s", first_members[:3])


def upload_unified_tar(
    s3_client,
    bucket: str,
    unified_key: str,
    tar_path: Path,
) -> None:
    """Upload unified tar to R2 using S3 Multipart Upload."""
    file_size = tar_path.stat().st_size
    config = TransferConfig(
        multipart_threshold=64 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        max_concurrency=8,
        use_threads=True,
    )

    pbar = None
    if tqdm is not None:
        pbar = tqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Uploading {tar_path.name}",
        )

        def callback(bytes_transferred):
            pbar.update(bytes_transferred)
    else:
        callback = None

    logger.info(
        "Uploading %s (%.2f GB) to s3://%s/%s via multipart upload...",
        tar_path,
        file_size / 1e9,
        bucket,
        unified_key,
    )
    s3_client.upload_file(
        Filename=str(tar_path),
        Bucket=bucket,
        Key=unified_key,
        Config=config,
        Callback=callback,
    )
    if pbar:
        pbar.close()


def verify_remote_object(s3_client, bucket: str, unified_key: str, expected_size: int) -> None:
    """Verify that the unified object exists in R2 and matches expected size."""
    resp = s3_client.head_object(Bucket=bucket, Key=unified_key)
    remote_size = resp.get("ContentLength", 0)
    if remote_size != expected_size:
        raise ValueError(
            f"Remote verification failed: remote size {remote_size} != local size {expected_size}"
        )
    logger.info(
        "Remote object verified: s3://%s/%s (%.2f GB, ETag: %s)",
        bucket,
        unified_key,
        remote_size / 1e9,
        resp.get("ETag", "N/A"),
    )


def delete_split_parts(s3_client, bucket: str, parts: list[dict]) -> int:
    """Delete split parts from R2 in batches of up to 1000."""
    if not parts:
        return 0

    deleted_count = 0
    batch_size = 1000
    for i in range(0, len(parts), batch_size):
        batch = parts[i : i + batch_size]
        delete_spec = {"Objects": [{"Key": p["Key"]} for p in batch]}
        resp = s3_client.delete_objects(Bucket=bucket, Delete=delete_spec)
        deleted_count += len(resp.get("Deleted", []))

    logger.info("Successfully deleted %d split part objects from s3://%s.", deleted_count, bucket)
    return deleted_count


def consolidate_archive(
    bucket: str = DEFAULT_BUCKET,
    unified_key: str = DEFAULT_KEY,
    work_dir: Path = Path("/content"),
    endpoint: str = DEFAULT_ENDPOINT,
    access_key: str | None = None,
    secret_key: str | None = None,
    dry_run: bool = False,
) -> None:
    """Main orchestration function to join parts and safely replace them on R2."""
    s3 = get_s3_client(endpoint, access_key, secret_key)

    # 1. Check free disk space
    free_bytes = shutil.disk_usage(work_dir).free
    free_gb = free_bytes / 1e9
    logger.info("Free disk at %s: %.1f GB", work_dir, free_gb)
    if free_gb < 18.0:
        raise RuntimeError(
            f"Insufficient disk space: {free_gb:.1f} GB available, at least 18 GB required."
        )

    # 2. Check if unified archive already exists
    try:
        head_resp = s3.head_object(Bucket=bucket, Key=unified_key)
        existing_size = head_resp.get("ContentLength", 0)
        logger.info(
            "Unified archive already exists on R2: s3://%s/%s (%.2f GB)",
            bucket, unified_key, existing_size / 1e9,
        )
    except Exception:
        head_resp = None

    # 3. List split parts
    parts = list_split_parts(s3, bucket, unified_key)
    if not parts:
        if head_resp is not None:
            logger.info("No split parts found; unified archive is already in place. Nothing to do!")
            return
        raise FileNotFoundError(
            f"No split parts matching prefix '{unified_key}.part_*' found in bucket '{bucket}'."
        )

    total_bytes = sum(p["Size"] for p in parts)
    logger.info(
        "Found %d split parts for %s (total: %.2f GB).",
        len(parts), unified_key, total_bytes / 1e9,
    )

    if dry_run:
        logger.info("[DRY RUN] Would download %d parts, upload unified tar, and delete parts.", len(parts))
        return

    # 4. Stream-download and join
    tar_path = work_dir / Path(unified_key).name
    logger.info("Streaming and concatenating %d parts to %s...", len(parts), tar_path)
    written_bytes = stream_download_parts(s3, bucket, parts, tar_path)
    logger.info("Local archive written: %.2f GB", written_bytes / 1e9)

    # 5. Verify local archive
    verify_tar_integrity(tar_path, total_bytes)

    # 6. Upload unified tar via S3 Multipart Upload
    upload_unified_tar(s3, bucket, unified_key, tar_path)

    # 7. Strictly verify remote object
    verify_remote_object(s3, bucket, unified_key, total_bytes)

    # 8. Safely delete split parts from R2
    logger.info("Safely deleting %d split parts from R2...", len(parts))
    delete_split_parts(s3, bucket, parts)

    # 9. Clean up local disk
    if tar_path.exists():
        tar_path.unlink()
        logger.info("Cleaned up local file %s.", tar_path)

    print("\n" + "=" * 60)
    print(f" SUCCESS: Unified archive s3://{bucket}/{unified_key} (%.2f GB)" % (total_bytes / 1e9))
    print(f" Replaced {len(parts)} split parts in R2.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Consolidate split R2 archive into a single .tar.")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help=f"R2 bucket (default: {DEFAULT_BUCKET})")
    parser.add_argument("--key", default=DEFAULT_KEY, help=f"Unified archive key (default: {DEFAULT_KEY})")
    parser.add_argument("--work-dir", type=Path, default=Path("/content"), help="Working directory (default: /content)")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="R2 S3 API endpoint")
    parser.add_argument("--dry-run", action="store_true", help="Inspect parts without modifying R2")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    consolidate_archive(
        bucket=args.bucket,
        unified_key=args.key,
        work_dir=args.work_dir,
        endpoint=args.endpoint,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
