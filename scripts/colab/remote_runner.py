"""Headless Colab remote runner for Stage 1 training with R2 sync and resume.

This script is executed on the remote Colab VM via `colab exec`.
It reads credentials from environment variables, sets up the environment,
resumes from R2 checkpoint if available, trains with live background checkpoint
sync, and uploads final artifacts to Cloudflare R2 upon completion.

Stage 1 v2 two-phase flow (docs/implementation_plan.md §4):
  phase 1 — main schedule from the base model (yolo26m on Colab L4);
  phase 2 — golden polish: short low-LR epochs over the phase-1 best with
            izutsumi slices oversampled 3x (CATNIP_POLISH=0 skips it; a
            polish failure falls back to uploading the phase-1 best under
            the same deliverable name).

The whole run profile is env-driven with v2 defaults; reproducing the v1
runner is purely a matter of exporting the v1 values, e.g.:
  CATNIP_STAGE1_BASE_MODEL_KEY=models/yolo26n.pt
  CATNIP_STAGE1_RUN_NAME=stage1 CATNIP_STAGE1_EPOCHS=100
  CATNIP_STAGE1_BATCH=64 CATNIP_POLISH=0
  CATNIP_STAGE1_ARTIFACT=yolo26_stage1_body_head_face.pt
"""

import os
import sys
import time
import shutil
import tarfile
import threading
from pathlib import Path

# --- Stage 1 v2 run profile (env-driven; v2 defaults) ---
STAGE1_BASE_MODEL_KEY = os.environ.get("CATNIP_STAGE1_BASE_MODEL_KEY", "models/yolo26m.pt")
STAGE1_RUN_NAME = os.environ.get("CATNIP_STAGE1_RUN_NAME", "stage1_v2")
STAGE1_EPOCHS = os.environ.get("CATNIP_STAGE1_EPOCHS", "75")
STAGE1_BATCH = os.environ.get("CATNIP_STAGE1_BATCH", "48")
STAGE1_DEVICE = os.environ.get("CATNIP_STAGE1_DEVICE", "cuda")
STAGE1_WORKERS = os.environ.get("CATNIP_STAGE1_WORKERS", "8")
POLISH_ENABLED = os.environ.get("CATNIP_POLISH", "1") != "0"
POLISH_EPOCHS = os.environ.get("CATNIP_POLISH_EPOCHS", "10")
STAGE1_ARTIFACT = os.environ.get("CATNIP_STAGE1_ARTIFACT", "yolo26m_stage1_v2.pt")
# Sliced-dataset archive key: the v2 profile needs the 3-class enriched
# dataset from plan task 3.4 (compressed/stage1_v2_sliced.tar). The archive
# is expected to extract to training/stage1_sliced/ (or pre-hydrate
# /content manually to that path). Until the v2 archive exists, explicitly
# point this back at compressed/stage1_sliced.tar — but note that trains
# v2 hyperparameters on partially labeled v1 slices.
STAGE1_DATASET_KEY = os.environ.get(
    "CATNIP_STAGE1_DATASET_KEY", "compressed/stage1_v2_sliced.tar"
)
CHECKPOINT_RESUME_KEY = f"runs/{STAGE1_RUN_NAME}-last.pt"

# --- 1. Validate Environment & Credentials ---
R2_ENDPOINT = os.environ.get(
    "R2_ENDPOINT",
    "https://eccb0b608bd926e793224944aba32cf4.r2.cloudflarestorage.com",
)
BUCKET = os.environ.get("BUCKET", "catnip-canonical")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY")

if not R2_ACCESS_KEY_ID or not R2_SECRET_ACCESS_KEY:
    raise ValueError(
        "Missing R2_ACCESS_KEY_ID or R2_SECRET_ACCESS_KEY environment variables."
    )

print(f"[RemoteRunner] Initializing on Colab VM...")
print(f"[RemoteRunner] Bucket: {BUCKET} | Endpoint: {R2_ENDPOINT}")
print(
    f"[RemoteRunner] Run profile: run={STAGE1_RUN_NAME}, base={STAGE1_BASE_MODEL_KEY}, "
    f"epochs={STAGE1_EPOCHS}, batch={STAGE1_BATCH}, device={STAGE1_DEVICE}, "
    f"workers={STAGE1_WORKERS}, polish={'on (' + POLISH_EPOCHS + ' epochs)' if POLISH_ENABLED else 'off'}, "
    f"artifact={STAGE1_ARTIFACT}"
)

# Install boto3 if missing
try:
    import boto3
except ImportError:
    print("[RemoteRunner] Installing boto3...")
    os.system("pip install --quiet boto3")
    import boto3

s3 = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    region_name="auto",
)

# Test R2 connectivity
try:
    resp = s3.list_objects_v2(Bucket=BUCKET, MaxKeys=1)
    print(f"[RemoteRunner] R2 bucket '{BUCKET}' verified successfully (KeyCount: {resp.get('KeyCount', 0)}).")
except Exception as e:
    raise ConnectionError(f"Failed to access R2 bucket '{BUCKET}': {e}")

# --- 2. Install Training Dependencies ---
print("[RemoteRunner] Verifying training dependencies...")
packages = [
    "ultralytics>=8.4.183,<9",
    "pydantic>=2.11,<3",
    "pydantic-settings>=2.11,<3",
    "omegaconf>=2.3,<3",
    "sahi>=0.11.36,<0.13",
    "opencv-python>=4.10,<5",
    "python-dotenv>=1,<2",
    "scikit-learn>=1.7,<2",
    "manga109api>=0.3.1,<0.4",
]
# NOTE: quoted with a pre-joined string — nested escaped quotes inside one
# f-string are Python 3.12-only syntax and this repo pins python 3.11.
pip_spec = " ".join(f'"{p}"' for p in packages)
os.system(f"pip install --quiet {pip_spec}")

# --- 3. Clone / Update Repository ---
REPO_URL = "https://github.com/rifusaki/catnip.git"
REPO_DIR = Path("/content/catnip")
# v2 lives on main (the yolo26m profile, polish mode, and the env-knob
# profile below). The historical default was the stale refactor/reID
# branch — pinning that silently trained pre-v2 code. Override with
# CATNIP_REPO_BRANCH for one-off runs against another branch.
BRANCH = os.environ.get("CATNIP_REPO_BRANCH", "main")

if not REPO_DIR.exists():
    print(f"[RemoteRunner] Cloning {REPO_URL} ({BRANCH})...")
    os.system(f"git clone -b {BRANCH} {REPO_URL} {REPO_DIR}")
else:
    print(f"[RemoteRunner] Updating repository at {REPO_DIR}...")
    os.system(f"cd {REPO_DIR} && git fetch && git checkout {BRANCH} && git pull")

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
os.chdir(REPO_DIR)

# --- 4. Setup Local NVMe Storage & Pull Dataset ---
LOCAL_DATA_ROOT = Path("/content/catnip-data-local")
LOCAL_DATA_ROOT.mkdir(parents=True, exist_ok=True)
(LOCAL_DATA_ROOT / "models").mkdir(parents=True, exist_ok=True)
os.environ["CATNIP_DATA"] = str(LOCAL_DATA_ROOT)

# The dataset dir follows the ARCHIVE KEY's stem, so the v2 archive
# (compressed/stage1_v2_sliced.tar -> training/stage1_v2_sliced/) needs
# no runner change; the v1 default keeps resolving training/stage1_sliced.
ARCHIVE_STEM = STAGE1_DATASET_KEY.rsplit("/", 1)[-1].removesuffix(".tar")
dataset_yaml = LOCAL_DATA_ROOT / "training" / ARCHIVE_STEM / "dataset.yaml"
if not dataset_yaml.exists():
    free_gb = shutil.disk_usage("/content").free / 1e9
    print(f"[RemoteRunner] /content free disk: {free_gb:.1f} GB")
    assert free_gb >= 35.0, f"Insufficient disk: {free_gb:.1f} GB < 35 GB required."

    tar_path = Path("/content/stage1_sliced.tar")
    try:
        s3.head_object(Bucket=BUCKET, Key=STAGE1_DATASET_KEY)
        archive_present = True
    except Exception:
        archive_present = False

    if archive_present:
        print(f"[RemoteRunner] Downloading dataset archive from R2: {STAGE1_DATASET_KEY}...")
        s3.download_file(BUCKET, STAGE1_DATASET_KEY, str(tar_path))
    else:
        # >5 GB archives cannot be stored as one R2 object (single-PUT cap);
        # the upload side splits them into ordered .part_* chunks. Rebuild.
        part_prefix = STAGE1_DATASET_KEY + ".part_"
        listing = s3.list_objects_v2(Bucket=BUCKET, Prefix=part_prefix)
        part_keys = sorted(obj["Key"] for obj in listing.get("Contents", []))
        if not part_keys:
            raise FileNotFoundError(
                f"Neither '{STAGE1_DATASET_KEY}' nor split parts '{part_prefix}*' "
                f"exist in bucket '{BUCKET}'."
            )
        print(f"[RemoteRunner] Archive stored as {len(part_keys)} split part(s); reconstructing...")
        part_paths = []
        for part_key in part_keys:
            part_local = Path("/content") / Path(part_key).name
            print(f"[RemoteRunner] Downloading {part_key}...")
            s3.download_file(BUCKET, part_key, str(part_local))
            part_paths.append(part_local)
        with open(tar_path, "wb") as out:
            for part in part_paths:
                with open(part, "rb") as chunk:
                    shutil.copyfileobj(chunk, out)
                part.unlink()
        print(f"[RemoteRunner] Reconstructed {tar_path} ({tar_path.stat().st_size/1e9:.1f} GB).")

    print("[RemoteRunner] Extracting dataset to local NVMe...")
    os.system(f"tar -xf {tar_path} -C {LOCAL_DATA_ROOT}")
    if tar_path.exists():
        tar_path.unlink()
    print("[RemoteRunner] Dataset extracted and archive cleaned up.")
else:
    print(
        f"[RemoteRunner] Using dataset already on disk at {dataset_yaml} "
        f"(expected archive key for fresh hydration: {STAGE1_DATASET_KEY})"
    )

# Patch dataset.yaml path to absolute path on Colab
import yaml as _yaml

data = _yaml.safe_load(dataset_yaml.read_text())
data["path"] = str(dataset_yaml.parent.resolve())
dataset_yaml.write_text(_yaml.dump(data, default_flow_style=False))
print(f"[RemoteRunner] Patched dataset.yaml path -> {data['path']}")

# --- 5. Inspect Checkpoint / Model Weights ---
resume_ckpt_path = None
# Check if R2 has a rolling last.pt for this run to resume from
try:
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=CHECKPOINT_RESUME_KEY, MaxKeys=1)
    if resp.get("KeyCount", 0) > 0:
        resume_ckpt_path = LOCAL_DATA_ROOT / "models" / f"{STAGE1_RUN_NAME}-resume.pt"
        print(f"[RemoteRunner] Found prior checkpoint in R2! Downloading {CHECKPOINT_RESUME_KEY}...")
        s3.download_file(BUCKET, CHECKPOINT_RESUME_KEY, str(resume_ckpt_path))
        print(f"[RemoteRunner] Checkpoint downloaded ({resume_ckpt_path.stat().st_size / 1e6:.1f} MB).")
except Exception as e:
    print(f"[RemoteRunner] Note: Could not check/download resume checkpoint: {e}")

if resume_ckpt_path is None or not resume_ckpt_path.exists():
    # Base weights are keyed by CATNIP_STAGE1_BASE_MODEL_KEY, not hardcoded.
    base_model_filename = Path(STAGE1_BASE_MODEL_KEY).name
    base_model_path = LOCAL_DATA_ROOT / "models" / base_model_filename
    if not base_model_path.exists():
        print(f"[RemoteRunner] Downloading base model weights from s3://{BUCKET}/{STAGE1_BASE_MODEL_KEY}...")
        s3.download_file(BUCKET, STAGE1_BASE_MODEL_KEY, str(base_model_path))
    model_arg = str(base_model_path)
    resume_flag = []
    print(f"[RemoteRunner] Starting fresh run from base weights: {model_arg}")
else:
    model_arg = str(resume_ckpt_path)
    resume_flag = ["--resume", str(resume_ckpt_path)]
    print(f"[RemoteRunner] Resuming training from checkpoint: {model_arg}")

# --- 6. Start Background Checkpoint Watcher ---
stop_watcher = threading.Event()

def checkpoint_watcher():
    last_pt = LOCAL_DATA_ROOT / "runs" / "detect" / STAGE1_RUN_NAME / "weights" / "last.pt"
    last_mtime = 0
    while not stop_watcher.is_set():
        if last_pt.exists():
            try:
                mtime = last_pt.stat().st_mtime
                if mtime > last_mtime:
                    s3.upload_file(str(last_pt), BUCKET, CHECKPOINT_RESUME_KEY)
                    print(f"\n[R2 Checkpoint] Pushed last.pt ({last_pt.stat().st_size / 1e6:.1f} MB) to s3://{BUCKET}/{CHECKPOINT_RESUME_KEY}")
                    last_mtime = mtime
            except Exception as e:
                print(f"\n[R2 Checkpoint Warning] Failed to push checkpoint: {e}")
        stop_watcher.wait(600)  # Check every 10 minutes

watcher_thread = threading.Thread(target=checkpoint_watcher, daemon=True)
watcher_thread.start()

# --- 7. Run Training (phase 1: main schedule) ---
phase1_overrides = [
    "--override", f"training.stage1.device={STAGE1_DEVICE}",
    "--override", f"training.stage1.workers={STAGE1_WORKERS}",
    "--override", f"training.stage1.batch={STAGE1_BATCH}",
    "--override", f"training.stage1.epochs={STAGE1_EPOCHS}",
    "--override", f"training.stage1.run_name={STAGE1_RUN_NAME}",
    "--override", "training.stage1.use_subset=false",
]
if resume_flag:
    # Chain resume: ultralytics restores full args from the checkpoint.
    phase1_argv = ["stage1.py", *phase1_overrides, *resume_flag, "--verbose"]
else:
    # Fresh start: pin the config model to the downloaded base weights
    # (config defaults to yolo26m; the env key owns the actual base so
    # v1 reproduction works from the same config).
    phase1_argv = [
        "stage1.py",
        "--override", f"training.stage1.model={model_arg}",
        *phase1_overrides,
        "--verbose",
    ]

orig_argv = sys.argv.copy()
sys.argv = phase1_argv
print(f"[RemoteRunner] Launching stage1.py (phase 1) with argv: {sys.argv}")
from scripts.train import stage1

try:
    stage1.main()
    print("\n[RemoteRunner] Phase 1 training completed successfully!")
except SystemExit as e:
    if e.code != 0:
        print(f"\n[RemoteRunner] Training exited with non-zero code {e.code}.")
        raise
finally:
    sys.argv = orig_argv

# --- 7b. Run Training (phase 2: golden polish) ---
phase1_best = LOCAL_DATA_ROOT / "runs" / "detect" / STAGE1_RUN_NAME / "weights" / "best.pt"
polish_best = LOCAL_DATA_ROOT / "runs" / "detect" / f"{STAGE1_RUN_NAME}_polish" / "weights" / "best.pt"

if POLISH_ENABLED and phase1_best.exists():
    # A stale best.pt from an EARLIER polish run on this VM must never ship
    # as the deliverable when the fresh polish aborts before its first save.
    if polish_best.exists():
        print(f"[RemoteRunner] Removing stale polish best.pt: {polish_best}")
        polish_best.unlink()
    sys.argv = [
        "stage1.py",
        "--polish",
        "--resume", str(phase1_best),
        "--override", f"training.stage1.device={STAGE1_DEVICE}",
        "--override", f"training.stage1.batch={STAGE1_BATCH}",
        "--override", f"training.stage1.epochs={POLISH_EPOCHS}",
        # polish derives "<run_name>_polish" from the config run_name; pin it
        # to the runner's run name so custom CATNIP_STAGE1_RUN_NAME values
        # write to (and are read from) runs/detect/<RUN_NAME>_polish.
        "--override", f"training.stage1.run_name={STAGE1_RUN_NAME}",
        "--verbose",
    ]
    print(f"\n[RemoteRunner] Launching golden polish (phase 2) with argv: {sys.argv}")
    try:
        stage1.main()
        print("\n[RemoteRunner] Golden polish (phase 2) completed successfully!")
    except SystemExit as e:
        if e.code == 0:
            print("\n[RemoteRunner] Golden polish (phase 2) completed.")
        else:
            print(f"\n[RemoteRunner] Golden polish exited with code {e.code}; falling back to the phase-1 best weights.")
    except Exception as e:
        print(f"\n[RemoteRunner] Golden polish failed ({e}); falling back to the phase-1 best weights.")
    finally:
        sys.argv = orig_argv
elif POLISH_ENABLED:
    print(f"\n[RemoteRunner] Golden polish skipped: no phase-1 best.pt at {phase1_best}.")

# --- 8. Final Artifacts Upload ---
stop_watcher.set()

runs_dir = LOCAL_DATA_ROOT / "runs" / "detect" / STAGE1_RUN_NAME
best_pt = LOCAL_DATA_ROOT / "models" / STAGE1_ARTIFACT

# The runner owns the deliverable choice: the polish best wins; if polish
# produced nothing (skipped/failed), the phase-1 best ships under the SAME
# deliverable name, with a log line recording which phase produced it.
if polish_best.exists():
    shutil.copy2(polish_best, best_pt)
    print(f"[RemoteRunner] Deliverable {STAGE1_ARTIFACT} produced by: golden polish (phase 2).")
elif (runs_dir / "weights" / "best.pt").exists():
    shutil.copy2(runs_dir / "weights" / "best.pt", best_pt)
    print(f"[RemoteRunner] Deliverable {STAGE1_ARTIFACT} produced by: main training (phase 1) — polish produced no best.pt.")
else:
    print(f"[RemoteRunner] WARNING: no best.pt in either phase run dir; cannot package {STAGE1_ARTIFACT}.")

if best_pt.exists():
    print(f"[RemoteRunner] Uploading best model to s3://{BUCKET}/models/{STAGE1_ARTIFACT}...")
    s3.upload_file(str(best_pt), BUCKET, f"models/{STAGE1_ARTIFACT}")

if runs_dir.exists():
    tar_out = Path("/content/stage1-run.tar")
    print(f"[RemoteRunner] Packaging run outputs to {tar_out}...")
    with tarfile.open(tar_out, "w") as tar:
        tar.add(str(runs_dir), arcname=STAGE1_RUN_NAME)
    print(f"[RemoteRunner] Uploading run archive to s3://{BUCKET}/runs/{STAGE1_RUN_NAME}-run.tar...")
    s3.upload_file(str(tar_out), BUCKET, f"runs/{STAGE1_RUN_NAME}-run.tar")

print("[RemoteRunner] All artifacts safely synchronized to R2. Done!")
