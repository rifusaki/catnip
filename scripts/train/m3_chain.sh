#!/usr/bin/env bash
# m3_chain.sh — chain one-epoch stage1_m3 processes around the MPS leak.
#
# Why: ultralytics-on-MPS (torch 2.6, macOS 27, M3 Air 16 GB) leaks Metal
# driver memory ~3-5 MB per training iteration — invisible to the torch
# allocator, unaffected by gc/empty_cache (see docs/stage1-runbook.md §E).
# A single process survives only ~1 epoch on the memory-safe subset before
# jetsam risk. So: run one epoch per process, kill right after the epoch's
# last.pt save, and resume the next epoch in a fresh process. Ultralytics
# resume restores epoch/optimizer/EMA/schedule — the cos_lr curve and
# early stopping behave exactly as one continuous run.
#
# The loop is checkpoint-driven: it reads the epoch stored in last.pt (0
# when absent), so it works from scratch AND resumes an interrupted chain.
#
# Usage:
#   pixi run train-stage1-m3-chain            # EPOCHS=40 default
#   EPOCHS=12 bash scripts/train/m3_chain.sh  # shorter run
#
# Tunables (env): EPOCHS (default 40), CYCLE_TIMEOUT_MIN (default 90).
# Each cycle: ~13 min train + ~2-3 min val/resume overhead on the M3 Air.
set -uo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO_ROOT"

EPOCHS="${EPOCHS:-40}"
CYCLE_TIMEOUT_MIN="${CYCLE_TIMEOUT_MIN:-90}"
RUN_DIR="${CATNIP_DATA:-/Volumes/rifuSSD/catnip-data}/runs/detect/stage1_m3"
LAST_PT="$RUN_DIR/weights/last.pt"

log() { printf '[chain %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

checkpoint_epoch() {
  # 0 when no checkpoint exists; else the epoch saved inside last.pt.
  if [ ! -f "$LAST_PT" ]; then
    echo 0
    return
  fi
  pixi run python -c "
import torch
print(torch.load('$LAST_PT', map_location='cpu', weights_only=False).get('epoch', 0))
" 2>/dev/null | tail -1
}

kill_cycle() {
  # Terminate the pixi wrapper, its python, and any stragglers.
  kill "$CAFFEINATE" 2>/dev/null
  kill "$WRAPPER" 2>/dev/null
  pkill -TERM -P "$WRAPPER" 2>/dev/null
  pkill -TERM -f "scripts/train/stage1.py" 2>/dev/null
  sleep 5
  pkill -KILL -f "scripts/train/stage1.py" 2>/dev/null
  wait "$WRAPPER" 2>/dev/null
}

while :; do
  DONE_EPOCH=$(checkpoint_epoch)
  case "$DONE_EPOCH" in
    ''|*[!0-9]*) DONE_EPOCH=0 ;;
  esac

  if [ "$DONE_EPOCH" -ge "$EPOCHS" ]; then
    log "checkpoint at epoch $DONE_EPOCH >= EPOCHS=$EPOCHS — chain complete"
    break
  fi

  RESUME_ARGS=()
  [ -f "$LAST_PT" ] && RESUME_ARGS=(--resume "$LAST_PT")
  NEXT_EPOCH=$((DONE_EPOCH + 1))
  log "training epoch $NEXT_EPOCH/$EPOCHS (checkpoint: $DONE_EPOCH)"

  BEFORE_MT=$(stat -f %m "$LAST_PT" 2>/dev/null || echo 0)
  pixi run train-stage1-m3 -- "${RESUME_ARGS[@]}" &
  WRAPPER=$!
  caffeinate -i -s -w "$WRAPPER" 2>/dev/null &
  CAFFEINATE=$!

  # The epoch that reaches EPOCHS runs to natural completion (final val +
  # best-model save) — do not kill it.
  if [ "$NEXT_EPOCH" -lt "$EPOCHS" ]; then
    DEADLINE=$((SECONDS + CYCLE_TIMEOUT_MIN * 60))
    SAVED=0
    while kill -0 "$WRAPPER" 2>/dev/null; do
      NOW_MT=$(stat -f %m "$LAST_PT" 2>/dev/null || echo 0)
      if [ "$NOW_MT" -gt "$BEFORE_MT" ]; then
        SAVED=1
        sleep 10  # let the checkpoint write fully flush
        break
      fi
      if [ "$SECONDS" -ge "$DEADLINE" ]; then
        log "no save within ${CYCLE_TIMEOUT_MIN}min — killing cycle"
        break
      fi
      sleep 10
    done
    kill_cycle
    if [ "$SAVED" -ne 1 ]; then
      log "epoch $NEXT_EPOCH ended without a save (crash/early-stop/resume failure)"
      log "inspect $RUN_DIR/results.csv; retry with: pixi run train-stage1-m3-chain"
      exit 1
    fi
    log "epoch $NEXT_EPOCH saved; memory released by process exit"
    sleep 15  # let macOS reclaim wired memory before the next cycle
  else
    wait "$WRAPPER"
    rc=$?
    log "final epoch $NEXT_EPOCH ran to completion (rc=$rc)"
    break
  fi
done

log "best weights: $RUN_DIR/weights/best.pt"
log "metrics: $RUN_DIR/results.csv"
