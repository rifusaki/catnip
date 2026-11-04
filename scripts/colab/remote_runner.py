"""Headless Colab remote runner for Stage 1 training with R2 sync and resume.

This script is executed on the remote Colab VM via `colab exec`.
It reads credentials from environment variables, sets up the environment,
resumes from R2 checkpoint if available, trains with live background checkpoint
sync, and uploads final artifacts to Cloudflare R2 upon completion.
"""

import os
import sys
import time
import shutil
import tarfile
import threading
from pathlib import Path

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
os.system(f"pip install --quiet {' '.join(f'\"{p}\"' for p in packages)}")

# --- 3. Clone / Update Repository ---
REPO_URL = "https://github.com/rifusaki/catnip.git"
REPO_DIR = Path("/content/catnip")
BRANCH = "refactor/reID"

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

dataset_yaml = LOCAL_DATA_ROOT / "training" / "stage1_sliced" / "dataset.yaml"
if not dataset_yaml.exists():
    free_gb = shutil.disk_usage("/content").free / 1e9
    print(f"[RemoteRunner] /content free disk: {free_gb:.1f} GB")
    assert free_gb >= 35.0, f"Insufficient disk: {free_gb:.1f} GB < 35 GB required."

    tar_path = Path("/content/stage1_sliced.tar")
    print("[RemoteRunner] Downloading dataset archive from R2 (14 GB)...")
    s3.download_file(BUCKET, "compressed/stage1_sliced.tar", str(tar_path))

    print("[RemoteRunner] Extracting dataset to local NVMe...")
    os.system(f"tar -xf {tar_path} -C {LOCAL_DATA_ROOT}")
    if tar_path.exists():
        tar_path.unlink()
    print("[RemoteRunner] Dataset extracted and archive cleaned up.")

# Patch dataset.yaml path to absolute path on Colab
import yaml as _yaml

data = _yaml.safe_load(dataset_yaml.read_text())
data["path"] = str(dataset_yaml.parent.resolve())
dataset_yaml.write_text(_yaml.dump(data, default_flow_style=False))
print(f"[RemoteRunner] Patched dataset.yaml path -> {data['path']}")

# --- 5. Inspect Checkpoint / Model Weights ---
resume_ckpt_path = None
# Check if R2 has runs/stage1-last.pt to resume from
try:
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix="runs/stage1-last.pt", MaxKeys=1)
    if resp.get("KeyCount", 0) > 0:
        resume_ckpt_path = LOCAL_DATA_ROOT / "models" / "stage1-resume.pt"
        print(f"[RemoteRunner] Found prior checkpoint in R2! Downloading runs/stage1-last.pt...")
        s3.download_file(BUCKET, "runs/stage1-last.pt", str(resume_ckpt_path))
        print(f"[RemoteRunner] Checkpoint downloaded ({resume_ckpt_path.stat().st_size / 1e6:.1f} MB).")
except Exception as e:
    print(f"[RemoteRunner] Note: Could not check/download resume checkpoint: {e}")

if resume_ckpt_path is None or not resume_ckpt_path.exists():
    base_model_path = LOCAL_DATA_ROOT / "models" / "yolo26n.pt"
    if not base_model_path.exists():
        print("[RemoteRunner] Downloading base model weights yolo26n.pt from R2...")
        s3.download_file(BUCKET, "models/yolo26n.pt", str(base_model_path))
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
    last_pt = LOCAL_DATA_ROOT / "runs" / "detect" / "stage1" / "weights" / "last.pt"
    last_mtime = 0
    while not stop_watcher.is_set():
        if last_pt.exists():
            try:
                mtime = last_pt.stat().st_mtime
                if mtime > last_mtime:
                    s3.upload_file(str(last_pt), BUCKET, "runs/stage1-last.pt")
                    print(f"\n[R2 Checkpoint] Pushed last.pt ({last_pt.stat().st_size / 1e6:.1f} MB) to s3://{BUCKET}/runs/stage1-last.pt")
                    last_mtime = mtime
            except Exception as e:
                print(f"\n[R2 Checkpoint Warning] Failed to push checkpoint: {e}")
        stop_watcher.wait(600)  # Check every 10 minutes

watcher_thread = threading.Thread(target=checkpoint_watcher, daemon=True)
watcher_thread.start()

# --- 7. Run Training ---
orig_argv = sys.argv.copy()
sys.argv = [
    "stage1.py",
    "--override", "training.stage1.device=cuda",
    "--override", "training.stage1.workers=8",
    "--override", "training.stage1.batch=64",
    "--override", "training.stage1.epochs=100",
    "--override", "training.stage1.use_subset=false",
    "--verbose",
] + resume_flag

print(f"[RemoteRunner] Launching stage1.py with argv: {sys.argv}")
from scripts.train import stage1

try:
    stage1.main()
    print("\n[RemoteRunner] Training completed successfully!")
except SystemExit as e:
    if e.code != 0:
        print(f"\n[RemoteRunner] Training exited with non-zero code {e.code}.")
        raise
finally:
    stop_watcher.set()
    sys.argv = orig_argv

# --- 8. Final Artifacts Upload ---
runs_dir = LOCAL_DATA_ROOT / "runs" / "detect" / "stage1"
best_pt = LOCAL_DATA_ROOT / "models" / "yolo26_stage1_body_head_face.pt"
if not best_pt.exists() and (runs_dir / "weights" / "best.pt").exists():
    shutil.copy2(runs_dir / "weights" / "best.pt", best_pt)

if best_pt.exists():
    print(f"[RemoteRunner] Uploading best model to s3://{BUCKET}/models/{best_pt.name}...")
    s3.upload_file(str(best_pt), BUCKET, f"models/{best_pt.name}")

if runs_dir.exists():
    tar_out = Path("/content/stage1-run.tar")
    print(f"[RemoteRunner] Packaging run outputs to {tar_out}...")
    with tarfile.open(tar_out, "w") as tar:
        tar.add(str(runs_dir), arcname="stage1")
    print(f"[RemoteRunner] Uploading run archive to s3://{BUCKET}/runs/stage1-run.tar...")
    s3.upload_file(str(tar_out), BUCKET, "runs/stage1-run.tar")

print("[RemoteRunner] All artifacts safely synchronized to R2. Done!")
