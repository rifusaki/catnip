#!/usr/bin/env bash
# launch_eval_360.sh — Run comparative Stage 1 evaluation headless from 360 WSL.
# Allocates/verifies the GPU session 'catnip-eval', injects R2 credentials,
# and executes scripts/colab/remote_eval_runner.py on the remote Colab VM.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
SESSION_NAME="catnip-eval"
EVAL_TIMEOUT_SECONDS="${CATNIP_EVAL_TIMEOUT_SECONDS:-21600}"

echo "=========================================================="
echo "  Catnip Evaluation — Headless Colab Runner (360 WSL)     "
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

COLAB_BIN="/root/.local/bin/colab"

if [[ ! -x "$COLAB_BIN" ]]; then
    COLAB_BIN="$(command -v colab || true)"
fi

if [[ -z "$COLAB_BIN" ]]; then
    echo "ERROR: colab CLI not found. Run: uv tool install google-colab-cli"
    exit 1
fi

echo "[1/4] Checking Colab session..."

# If the training session is still lingering, stop it to prevent Colab GPU concurrency conflicts
if "$COLAB_BIN" sessions 2>/dev/null | grep -q "catnip-run"; then
    echo "Detected existing training session 'catnip-run'."
    echo "Stopping 'catnip-run' to ensure Colab assigns GPU quota to evaluation..."
    "$COLAB_BIN" stop -s "catnip-run" 2>/dev/null || true
fi

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
PAYLOAD_FILE=$(mktemp /tmp/remote_eval_payload.XXXXXX.py)
trap 'rm -f "$PAYLOAD_FILE"' EXIT

cat << EOF > "$PAYLOAD_FILE"
import os
os.environ["R2_ENDPOINT"] = "${R2_ENDPOINT}"
os.environ["BUCKET"] = "${BUCKET}"
os.environ["R2_ACCESS_KEY_ID"] = "${R2_ACCESS_KEY_ID}"
os.environ["R2_SECRET_ACCESS_KEY"] = "${R2_SECRET_ACCESS_KEY}"
os.environ["CATNIP_EVAL_SAMPLE"] = "${CATNIP_EVAL_SAMPLE:-0}"
os.environ["CATNIP_EVAL_DEVICE"] = "${CATNIP_EVAL_DEVICE:-cuda}"
EOF

# Forward the v2 benchmark knobs when set here (defaults keep the classic
# v1 flow: the fixed harness re-benchmarks the v1 deliverable into the
# canonical results/benchmarks prefix — task 2.1).
for _knob in \
    CATNIP_REPO_BRANCH \
    CATNIP_EVAL_MODEL_KEY \
    CATNIP_EVAL_RESULTS_PREFIX; do
    if [[ -n "${!_knob:-}" ]]; then
        printf 'os.environ["%s"] = "%s"\n' "$_knob" "${!_knob}" >> "$PAYLOAD_FILE"
    fi
done

cat "${SCRIPT_DIR}/remote_eval_runner.py" >> "$PAYLOAD_FILE"

echo "[4/4] Starting evaluation stream on Colab VM..."
echo "----------------------------------------------------------"
echo "  TIP: If running in tmux, press 'Ctrl+B' then 'D' to detach."
echo "  Reattach anytime with: tmux attach -t catnip-eval"
echo "  Overrides: CATNIP_EVAL_SAMPLE (0 = full split), CATNIP_EVAL_DEVICE,"
echo "             CATNIP_EVAL_TIMEOUT_SECONDS (default 21600)."
echo "  v2 (task 4.3): CATNIP_EVAL_MODEL_KEY=models/yolo26m_stage1_v2.pt and"
echo "             CATNIP_EVAL_RESULTS_PREFIX=results/benchmarks_v2 keep the"
echo "             canonical v1 results intact; CATNIP_REPO_BRANCH (default main)."
echo "----------------------------------------------------------"

"$COLAB_BIN" exec -s "$SESSION_NAME" --timeout "$EVAL_TIMEOUT_SECONDS" -f "$PAYLOAD_FILE"

echo "=========================================================="
echo "  Evaluation finished! Results synced to                  "
echo "  s3://catnip-canonical/results/benchmarks/               "
echo "=========================================================="
