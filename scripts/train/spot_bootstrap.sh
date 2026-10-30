#!/usr/bin/env bash
# Stage 1 training bootstrap for spot GPU VMs (Vast.ai / RunPod / any CUDA host).
#
# Pulls the sliced dataset from R2 (zero egress), repoints dataset.yaml to the
# local data root, trains with periodic checkpoint pushes back to R2, and
# publishes the deliverable model. Interruption-safe: re-run the same command
# after a preemption to resume from the last pushed checkpoint.
#
# Required environment (or file via R2_CREDS, e.g. ~/secrets/r2-sippy.conf):
#   AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY   (R2 token)
#     - the dev-machine conf uses R2_ACCESS_KEY_ID/R2_SECRET_ACCESS_KEY;
#       this script maps them automatically
# Optional overrides:
#   CATNIP_REPO     repo checkout        (default ~/catnip; cloned if missing)
#   CATNIP_DATA     data root           (default ~/catnip-data)
#   R2_ENDPOINT     S3 endpoint          (default catnip-canonical account)
#   SPOT_BATCH      train batch         (default 32)
#   SPOT_WORKERS    dataloader workers  (default 8)
#   CKPT_INTERVAL_MIN  checkpoint push cadence (default 10)
set -euo pipefail

CATNIP_REPO="${CATNIP_REPO:-$HOME/catnip}"
CATNIP_DATA="${CATNIP_DATA:-$HOME/catnip-data}"
R2_ENDPOINT="${R2_ENDPOINT:-https://eccb0b608bd926e793224944aba32cf4.r2.cloudflarestorage.com}"
R2_BUCKET="${R2_BUCKET:-catnip-canonical}"
SPOT_BATCH="${SPOT_BATCH:-32}"
SPOT_WORKERS="${SPOT_WORKERS:-8}"
CKPT_INTERVAL_MIN="${CKPT_INTERVAL_MIN:-10}"

log() { printf '[spot] %s\n' "$*"; }
die() { printf '[spot] ERROR: %s\n' "$*" >&2; exit 1; }

# --- R2 credentials ---------------------------------------------------------
if [[ -n "${R2_CREDS:-}" && -f "${R2_CREDS}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${R2_CREDS}"
  set +a
fi
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-${R2_ACCESS_KEY_ID:-}}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-${R2_SECRET_ACCESS_KEY:-}}"
export AWS_REGION="${AWS_REGION:-auto}"
[[ -n "$AWS_ACCESS_KEY_ID" && -n "$AWS_SECRET_ACCESS_KEY" ]] \
  || die "R2 credentials missing. Set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or R2_CREDS (conf file with R2_ACCESS_KEY_ID/R2_SECRET_ACCESS_KEY)."
[[ -n "${AWS_REGION}" ]] || export AWS_REGION=auto

# --- R2 transfer helper: s5cmd if available, else boto3 ----------------------
r2_get() {  # r2_get <s3-url> <local-path>
  if command -v s5cmd >/dev/null 2>&1; then
    s5cmd --endpoint-url "$R2_ENDPOINT" cp "$1" "$2"
  else
    python3 - "$1" "$2" <<'PY'
import sys
import boto3
src, dst = sys.argv[1], sys.argv[2]
bucket, key = src.removeprefix("s3://").split("/", 1)
boto3.client("s3").download_file(bucket, key, dst)
PY
  fi
}
r2_put() {  # r2_put <local-path> <s3-url>
  if command -v s5cmd >/dev/null 2>&1; then
    s5cmd --endpoint-url "$R2_ENDPOINT" cp "$1" "$2"
  else
    python3 - "$1" "$2" <<'PY'
import sys
import boto3
src, dst = sys.argv[1], sys.argv[2]
bucket, key = dst.removeprefix("s3://").split("/", 1)
boto3.client("s3").upload_file(src, bucket, key)
PY
  fi
}

# --- Preflight ---------------------------------------------------------------
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \
  || log "WARNING: nvidia-smi not found — no GPU visible?"
free_disk=$(df -Pk "$CATNIP_DATA" 2>/dev/null | awk 'NR==2 {print $4}')
(( free_disk >= 40 * 1024 * 1024 )) 2>/dev/null \
  || { mkdir -p "$CATNIP_DATA"; free_disk=$(df -Pk "$CATNIP_DATA" | awk 'NR==2 {print $4}'); }
(( free_disk >= 40 * 1024 * 1024 )) \
  || die "Less than 40 GB free under $CATNIP_DATA (need ~14 GB tarball + ~14 GB extracted + cache)."

if ! command -v s5cmd >/dev/null 2>&1; then
  python3 -c "import boto3" 2>/dev/null || pip install --quiet boto3 \
    || die "Need s5cmd or python3+boto3 for R2 transfers."
fi

# --- Repo + env --------------------------------------------------------------
if [[ ! -d "$CATNIP_REPO/.git" ]]; then
  log "Cloning catnip into $CATNIP_REPO"
  git clone https://github.com/rifusaki/catnip.git "$CATNIP_REPO"
fi
cd "$CATNIP_REPO"

export CATNIP_DATA
command -v pixi >/dev/null 2>&1 || curl -fsSL https://pixi.sh/install.sh | sh
export PATH="$HOME/.pixi/bin:$PATH"
log "Installing cuda env (conda pytorch, CUDA 12.6)"
pixi install -e cuda
pixi run -e cuda python -c "import torch; print('cuda available:', torch.cuda.is_available())"

# --- Dataset ingest from R2 (zero egress) ------------------------------------
SLICED="$CATNIP_DATA/training/stage1_sliced"
if [[ ! -f "$SLICED/dataset.yaml" ]]; then
  log "Pulling stage1_sliced.tar (14 GB) from R2"
  r2_get "s3://$R2_BUCKET/compressed/stage1_sliced.tar" "$CATNIP_DATA/stage1_sliced.tar"
  tar -xf "$CATNIP_DATA/stage1_sliced.tar" -C "$CATNIP_DATA"
  rm "$CATNIP_DATA/stage1_sliced.tar"
fi
# dataset.yaml ships with the generating host's absolute `path:` — repoint it.
sed -i "s|^path:.*|path: $SLICED|" "$SLICED/dataset.yaml"
log "dataset.yaml path -> $SLICED"

mkdir -p "$CATNIP_DATA/models"
[[ -f "$CATNIP_DATA/models/yolo26n.pt" ]] \
  || r2_get "s3://$R2_BUCKET/models/yolo26n.pt" "$CATNIP_DATA/models/yolo26n.pt"

# --- Resume: hydrate checkpoint from R2 if present ----------------------------
RUN_DIR="$CATNIP_DATA/runs/detect/stage1"
RESUME_ARGS=()
if [[ ! -f "$RUN_DIR/weights/last.pt" ]]; then
  if r2_get "s3://$R2_BUCKET/runs/stage1-last.pt" "$CATNIP_DATA/stage1-last.pt.part" 2>/dev/null; then
    mkdir -p "$RUN_DIR/weights"
    mv "$CATNIP_DATA/stage1-last.pt.part" "$RUN_DIR/weights/last.pt"
    log "Resumed checkpoint hydrated from R2"
  fi
fi
[[ -f "$RUN_DIR/weights/last.pt" ]] && RESUME_ARGS=(--resume "$RUN_DIR/weights/last.pt")

# --- Checkpoint watcher: push last.pt to R2 every CKPT_INTERVAL_MIN ----------
watcher() {
  while kill -0 "$1" 2>/dev/null; do
    sleep "${CKPT_INTERVAL_MIN}m"
    [[ -f "$RUN_DIR/weights/last.pt" ]] || continue
    r2_put "$RUN_DIR/weights/last.pt" "s3://$R2_BUCKET/runs/stage1-last.pt" \
      && log "checkpoint pushed ($(date -u +%H:%M:%SZ))"
  done
}
pixi run -e cuda train-stage1 \
  --override training.stage1.device=cuda \
              training.stage1.batch="$SPOT_BATCH" \
              training.stage1.workers="$SPOT_WORKERS" \
  "${RESUME_ARGS[@]}" &
TRAIN_PID=$!
watcher "$TRAIN_PID" &
WATCHER_PID=$!
trap 'kill "$WATCHER_PID" 2>/dev/null || true' EXIT
wait "$TRAIN_PID"

# --- Publish results back to R2 ----------------------------------------------
r2_put "$CATNIP_DATA/models/yolo26_stage1_body_head_face.pt" \
  "s3://$R2_BUCKET/models/yolo26_stage1_body_head_face.pt"
tar -C "$CATNIP_DATA" -cf "$CATNIP_DATA/stage1-run.tar" runs/detect/stage1
r2_put "$CATNIP_DATA/stage1-run.tar" "s3://$R2_BUCKET/runs/stage1-run.tar"
rm "$CATNIP_DATA/stage1-run.tar"
r2_put "$RUN_DIR/weights/last.pt" "s3://$R2_BUCKET/runs/stage1-last.pt" || true
log "Done. Deliverable: s3://$R2_BUCKET/models/yolo26_stage1_body_head_face.pt"
