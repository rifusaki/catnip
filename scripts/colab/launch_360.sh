#!/usr/bin/env bash
# launch_360.sh — Run Stage 1 Colab training 24/7 from 360 WSL server.
# Keeps Colab actively connected regardless of your laptop sleeping.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

echo "=========================================================="
echo "  Catnip Stage 1 — 24/7 Headless Colab Runner (360 WSL)   "
echo "=========================================================="

# Check credentials file
if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: Missing ${ENV_FILE}"
    echo "Please copy ${SCRIPT_DIR}/.env.template to ${ENV_FILE} and fill in your R2 credentials."
    exit 1
fi

# Export environment variables from .env
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

if [[ -z "${R2_ACCESS_KEY_ID:-}" || "${R2_ACCESS_KEY_ID:-}" == "INSERT_YOUR_R2_ACCESS_KEY_ID_HERE" ]]; then
    echo "ERROR: R2_ACCESS_KEY_ID is not configured in ${ENV_FILE}."
    exit 1
fi
if [[ -z "${R2_SECRET_ACCESS_KEY:-}" || "${R2_SECRET_ACCESS_KEY:-}" == "INSERT_YOUR_R2_SECRET_ACCESS_KEY_HERE" ]]; then
    echo "ERROR: R2_SECRET_ACCESS_KEY is not configured in ${ENV_FILE}."
    exit 1
fi

SESSION_NAME="catnip-run"
COLAB_BIN="/root/.local/bin/colab"

if [[ ! -x "$COLAB_BIN" ]]; then
    COLAB_BIN="$(command -v colab || true)"
fi

if [[ -z "$COLAB_BIN" ]]; then
    echo "ERROR: colab CLI not found. Run: uv tool install google-colab-cli"
    exit 1
fi

echo "[1/4] Checking Colab session..."

session_exists=false
if "$COLAB_BIN" sessions 2>/dev/null | grep -q "$SESSION_NAME"; then
    session_exists=true
fi

# Verify if the existing session is truly responsive and alive
session_alive=false
if [[ "$session_exists" == "true" ]]; then
    status_output=$("$COLAB_BIN" status -s "$SESSION_NAME" 2>&1 || true)
    if echo "$status_output" | grep -Eq "IDLE|BUSY"; then
        session_alive=true
        echo "Reusing active, healthy session: '${SESSION_NAME}'"
    else
        echo "Session '${SESSION_NAME}' is present in local cache but remote runtime is unreachable/terminated."
        echo "Pruning stale session from Colab CLI state..."
        "$COLAB_BIN" stop -s "$SESSION_NAME" 2>/dev/null || true
    fi
fi

if [[ "$session_alive" != "true" ]]; then
    echo "Allocating fresh L4 GPU session on Colab: '${SESSION_NAME}'..."
    "$COLAB_BIN" new -s "$SESSION_NAME" --gpu L4
    echo "Session allocated! Waiting for kernel initialization..."
    for i in {1..30}; do
        if "$COLAB_BIN" status -s "$SESSION_NAME" 2>/dev/null | grep -q "IDLE"; then
            echo "Kernel is READY."
            break
        fi
        sleep 2
    done
fi

echo "[2/4] Verifying GPU status on remote VM..."
echo '!nvidia-smi' | "$COLAB_BIN" exec -s "$SESSION_NAME" || true

echo "[3/4] Preparing remote payload with R2 credentials..."
PAYLOAD_FILE=$(mktemp /tmp/remote_runner_payload.XXXXXX.py)
trap 'rm -f "$PAYLOAD_FILE"' EXIT

cat << EOF > "$PAYLOAD_FILE"
import os
os.environ["R2_ENDPOINT"] = "${R2_ENDPOINT}"
os.environ["BUCKET"] = "${BUCKET}"
os.environ["R2_ACCESS_KEY_ID"] = "${R2_ACCESS_KEY_ID}"
os.environ["R2_SECRET_ACCESS_KEY"] = "${R2_SECRET_ACCESS_KEY}"
EOF

# Forward the v2 training profile knobs when they are set on this WSL
# host (the runner's in-code v2 defaults apply otherwise). Values are
# ops strings (branch names, R2 keys, ints) — no quoting gymnastics.
for _knob in \
    CATNIP_REPO_BRANCH \
    CATNIP_STAGE1_BASE_MODEL_KEY \
    CATNIP_STAGE1_RUN_NAME \
    CATNIP_STAGE1_EPOCHS \
    CATNIP_STAGE1_BATCH \
    CATNIP_STAGE1_DEVICE \
    CATNIP_STAGE1_WORKERS \
    CATNIP_STAGE1_DATASET_KEY \
    CATNIP_STAGE1_ARTIFACT \
    CATNIP_POLISH \
    CATNIP_POLISH_EPOCHS; do
    if [[ -n "${!_knob:-}" ]]; then
        printf 'os.environ["%s"] = "%s"\n' "$_knob" "${!_knob}" >> "$PAYLOAD_FILE"
    fi
done

cat "${SCRIPT_DIR}/remote_runner.py" >> "$PAYLOAD_FILE"

echo "[4/4] Starting training stream on Colab VM..."
echo "----------------------------------------------------------"
echo "  TIP: If running in tmux, press 'Ctrl+B' then 'D' to detach."
echo "  Reattach anytime with: tmux attach -t catnip-training"
echo "  v2 profile knobs (forwarded when set): CATNIP_REPO_BRANCH (default"
echo "  main), CATNIP_STAGE1_{BASE_MODEL_KEY,RUN_NAME,EPOCHS,BATCH,DEVICE,"
echo "  WORKERS,DATASET_KEY,ARTIFACT}, CATNIP_POLISH, CATNIP_POLISH_EPOCHS."
echo "  Fresh v2 runs need models/yolo26m.pt and the"
echo "  CATNIP_STAGE1_DATASET_KEY archive present in R2."
echo "----------------------------------------------------------"

"$COLAB_BIN" exec -s "$SESSION_NAME" --timeout 86400 -f "$PAYLOAD_FILE"

echo "=========================================================="
echo "  Training process finished! Check R2 for saved weights.  "
echo "=========================================================="
