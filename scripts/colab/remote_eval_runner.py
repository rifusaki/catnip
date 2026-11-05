"""Headless Colab remote runner for comparative Stage 1 evaluation.

Executed on the remote Colab VM via `colab exec` (launched from `360 WSL` by
`scripts/colab/launch_eval_360.sh`). Pulls the sliced test dataset and the
trained Stage 1 model from Cloudflare R2, runs the comparative face and head
benchmarks (`scripts/eval/benchmark_faces.py`), and synchronizes every report
(`summary.md`, `metrics.json`) and visual panel (`visuals/*.jpg`) back to
`s3://catnip-canonical/results/benchmarks/{face,head}/`.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# --- 1. Validate Environment & Credentials ---
R2_ENDPOINT = os.environ.get(
    "R2_ENDPOINT",
    "https://eccb0b608bd926e793224944aba32cf4.r2.cloudflarestorage.com",
)
BUCKET = os.environ.get("BUCKET", "catnip-canonical")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY")
DATASET_ARCHIVE_KEY = "compressed/stage1_sliced.tar"
TRAINED_MODEL_KEY = "models/yolo26_stage1_body_head_face.pt"
RESULTS_PREFIX = "results/benchmarks"
EVAL_SAMPLE = os.environ.get("CATNIP_EVAL_SAMPLE", "0")
EVAL_DEVICE = os.environ.get("CATNIP_EVAL_DEVICE", "cuda")

if os.environ.get("CUDA_VISIBLE_DEVICES") == "cuda":
    del os.environ["CUDA_VISIBLE_DEVICES"]

if not R2_ACCESS_KEY_ID or not R2_SECRET_ACCESS_KEY:
    raise ValueError(
        "Missing R2_ACCESS_KEY_ID or R2_SECRET_ACCESS_KEY environment variables."
    )

print("[RemoteEval] Initializing on Colab VM...")
print(f"[RemoteEval] Bucket: {BUCKET} | Endpoint: {R2_ENDPOINT}")

try:
    import boto3
except ImportError:
    print("[RemoteEval] Installing boto3...")
    os.system("pip install --quiet boto3")
    import boto3

s3 = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    region_name="auto",  # required by boto3, ignored by R2
)

try:
    s3.head_bucket(Bucket=BUCKET)
except Exception as err:
    raise ConnectionError(f"Failed to access R2 bucket '{BUCKET}': {err}") from err
print(f"[RemoteEval] R2 bucket '{BUCKET}' verified successfully.")

# --- 2. Install Benchmark Dependencies ---
print("[RemoteEval] Verifying benchmark dependencies...")
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
# The benchmark suite (scripts/eval/, src/eval/) lives on main; the older
# refactor/reID branch does not contain it.
REPO_URL = "https://github.com/rifusaki/catnip.git"
REPO_DIR = Path("/content/catnip")
BRANCH = "main"

if not REPO_DIR.exists():
    print(f"[RemoteEval] Cloning {REPO_URL} ({BRANCH})...")
    subprocess.run(["git", "clone", "-b", BRANCH, REPO_URL, str(REPO_DIR)], check=True)
else:
    # The VM may hold a checkout from a different payload (e.g. the training
    # runner), so enforce the branch before pulling.
    print(f"[RemoteEval] Updating repository at {REPO_DIR}...")
    subprocess.run(["git", "-C", str(REPO_DIR), "fetch"], check=True)
    subprocess.run(["git", "-C", str(REPO_DIR), "checkout", BRANCH], check=True)
    subprocess.run(["git", "-C", str(REPO_DIR), "pull"], check=True)

benchmark_script = REPO_DIR / "scripts" / "eval" / "benchmark_faces.py"
if not benchmark_script.exists():
    raise FileNotFoundError(
        f"{benchmark_script} missing after clone; is BRANCH='{BRANCH}' up to date?"
    )

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
os.chdir(REPO_DIR)

# --- 4. Pull Dataset & Trained Model from R2 ---
LOCAL_DATA_ROOT = Path("/content/catnip-data-local")
(LOCAL_DATA_ROOT / "models").mkdir(parents=True, exist_ok=True)
os.environ["CATNIP_DATA"] = str(LOCAL_DATA_ROOT)

dataset_yaml = LOCAL_DATA_ROOT / "training" / "stage1_sliced" / "dataset.yaml"
if not dataset_yaml.exists():
    free_gb = shutil.disk_usage("/content").free / 1e9
    print(f"[RemoteEval] /content free disk: {free_gb:.1f} GB")
    assert free_gb >= 35.0, f"Insufficient disk: {free_gb:.1f} GB < 35 GB required."

    tar_path = Path("/content/stage1_sliced.tar")
    print(f"[RemoteEval] Downloading dataset archive {DATASET_ARCHIVE_KEY} from R2 (14 GB)...")
    try:
        s3.download_file(BUCKET, DATASET_ARCHIVE_KEY, str(tar_path))
    except Exception as err:
        raise RuntimeError(
            f"Could not download '{DATASET_ARCHIVE_KEY}' from R2 bucket '{BUCKET}': {err}"
        ) from err

    print("[RemoteEval] Extracting dataset to local NVMe...")
    os.system(f"tar -xf {tar_path} -C {LOCAL_DATA_ROOT}")
    if tar_path.exists():
        tar_path.unlink()
    print("[RemoteEval] Dataset extracted and archive cleaned up.")

assert dataset_yaml.exists(), f"Extraction error: {dataset_yaml} does not exist."

trained_model = LOCAL_DATA_ROOT / "models" / "yolo26_stage1_body_head_face.pt"
if not trained_model.exists():
    print(f"[RemoteEval] Downloading trained model {TRAINED_MODEL_KEY} from R2...")
    try:
        s3.download_file(BUCKET, TRAINED_MODEL_KEY, str(trained_model))
    except Exception as err:
        raise RuntimeError(
            f"Could not download '{TRAINED_MODEL_KEY}' from R2 bucket '{BUCKET}': {err}"
        ) from err

assert trained_model.exists(), f"Model fetch error: {trained_model} does not exist."
print(f"[RemoteEval] Trained model ready ({trained_model.stat().st_size / 1e6:.1f} MB).")

# Patch dataset.yaml to the absolute Colab path (required by the benchmark harness)
import yaml as _yaml

data = _yaml.safe_load(dataset_yaml.read_text())
data["path"] = str(dataset_yaml.parent.resolve())
dataset_yaml.write_text(_yaml.dump(data, default_flow_style=False))
print(f"[RemoteEval] Patched dataset.yaml path -> {data['path']}")

# --- 5. Run Benchmarks & Sync Results to R2 ---
print(f"[RemoteEval] Sample size: {EVAL_SAMPLE} (0 = full test split) | Device: {EVAL_DEVICE}")

EVAL_TARGETS = [
    ("face", "catnip,fuyucchi_face,zymk9_face,deepghs_face_nano"),
    ("head", "catnip,deepghs_head_nano"),
]


def list_remote_keys(prefix: str) -> list[str]:
    """List all R2 object keys under a prefix (paginated)."""
    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        keys.extend(obj["Key"] for obj in page.get("Contents", []))
    return keys


def sync_results(target: str, output_dir: Path, prune_visuals: bool = False) -> int:
    """Upload this run's summary.md, metrics.json, and visuals/*.jpg to results/benchmarks/<target>/.

    The output directory is recreated fresh before every benchmark run, so
    every artifact found here belongs to the current run, never a previous one.
    Returns the number of uploaded artifacts.
    """
    if not output_dir.exists():
        print(f"[RemoteEval][{target}] no output directory at {output_dir}; nothing to sync.")
        return 0
    prefix = f"{RESULTS_PREFIX}/{target}"
    uploaded = 0
    for filename in ("summary.md", "metrics.json"):
        report = output_dir / filename
        if report.exists():
            s3.upload_file(str(report), BUCKET, f"{prefix}/{filename}")
            uploaded += 1
        else:
            print(f"[RemoteEval][{target}] missing {filename}; benchmark output incomplete.")
    visuals_dir = output_dir / "visuals"
    panels = sorted(visuals_dir.glob("*.jpg")) if visuals_dir.is_dir() else []
    if panels and prune_visuals:
        stale_panels = list_remote_keys(f"{prefix}/visuals/")
        for i in range(0, len(stale_panels), 1000):
            s3.delete_objects(
                Bucket=BUCKET,
                Delete={"Objects": [{"Key": k} for k in stale_panels[i:i + 1000]], "Quiet": True},
            )
        if stale_panels:
            print(f"[RemoteEval][{target}] pruned {len(stale_panels)} stale visual panel(s) from s3://{BUCKET}/{prefix}/visuals/")
    for panel in panels:
        s3.upload_file(str(panel), BUCKET, f"{prefix}/visuals/{panel.name}")
        uploaded += 1
    print(
        f"[RemoteEval][{target}] synced {uploaded} artifact(s), {len(panels)} visual panels "
        f"to s3://{BUCKET}/{prefix}/"
    )
    return uploaded


for target, models in EVAL_TARGETS:
    output_dir = LOCAL_DATA_ROOT / "results" / f"benchmarks_{target}"

    # Fresh output directory per run: guarantees synced artifacts always
    # belong to the current run, never to a previous one.
    shutil.rmtree(output_dir, ignore_errors=True)

    cmd = [
        sys.executable, str(benchmark_script),
        "--target", target,
        "--models", models,
        "--catnip-model", str(trained_model),
        "--dataset-yaml", str(dataset_yaml),
        "--device", EVAL_DEVICE,
        "--sample", EVAL_SAMPLE,
        "--output-dir", str(output_dir),
    ]
    print(f"[RemoteEval] Launching {target} benchmark: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as err:
        partial = sync_results(target, output_dir)
        raise RuntimeError(
            f"{target} benchmark failed with exit code {err.returncode}; "
            f"{partial} partial artifact(s) from this run synced to s3://{BUCKET}/{RESULTS_PREFIX}/{target}/."
        ) from err
    sync_results(target, output_dir, prune_visuals=True)

print(f"[RemoteEval] All benchmark artifacts synchronized to s3://{BUCKET}/{RESULTS_PREFIX}/. Done!")
