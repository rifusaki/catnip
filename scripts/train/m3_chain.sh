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
# NOTE: ultralytics check_resume restores ALL args from the checkpoint —
# epochs cannot be overridden on resume — so the first cycle must be
# launched with the intended EPOCHS, and the chain refuses to continue a
# checkpoint saved with different epochs.
#
# Usage:
#   pixi run train-stage1-m3-chain            # EPOCHS=26 default (fits 12h)
#   EPOCHS=12 bash scripts/train/m3_chain.sh  # shorter run
#
# Tunables (env): EPOCHS (default 26), CYCLE_TIMEOUT_MIN (default 90),
# MAX_WALL_CLOCK_HOURS (default 12).
# Sustained cycle on fanless M3 Air: ~26 min (train + val + thermal throttle + flush).
set -uo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO_ROOT"

EPOCHS="${EPOCHS:-26}"
CYCLE_TIMEOUT_MIN="${CYCLE_TIMEOUT_MIN:-90}"
MAX_WALL_CLOCK_HOURS="${MAX_WALL_CLOCK_HOURS:-12}"
RUN_DIR="${CATNIP_DATA:-/Volumes/rifuSSD/catnip-data}/runs/detect/stage1_m3"
LAST_PT="$RUN_DIR/weights/last.pt"
CHAIN_START_TIME=$SECONDS

log() { printf '[chain %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

checkpoint_state() {
  # "<epoch> <saved_epochs_arg>" — epoch -1 is ultralytics' completed
  # sentinel; the saved epochs arg must match EPOCHS or resume would
  # train the wrong schedule (check_resume restores args from the
  # checkpoint and does NOT allow overriding epochs).
  if [ ! -f "$LAST_PT" ]; then
    echo "0 none"
    return
  fi
  pixi run python -c "
import torch
ck = torch.load('$LAST_PT', map_location='cpu', weights_only=False)
print(ck.get('epoch', 0), ck.get('train_args', {}).get('epochs'))
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
  read -r DONE_EPOCH SAVED_EPOCHS <<< "$(checkpoint_state)"
  case "$DONE_EPOCH" in
    -1)
      log "checkpoint marks training complete (epoch -1) — chain done"
      break
      ;;
    ''|*[!0-9]*)
      DONE_EPOCH=0
      ;;
  esac

  if [ "$SAVED_EPOCHS" != "none" ] && [ "$SAVED_EPOCHS" != "$EPOCHS" ]; then
    log "checkpoint's saved epochs=$SAVED_EPOCHS != EPOCHS=$EPOCHS —"
    log "ultralytics resume restores args from the checkpoint and ignores"
    log "an epochs override; delete $RUN_DIR and start the chain fresh."
    exit 1
  fi

  if [ "$DONE_EPOCH" -ge "$EPOCHS" ]; then
    log "checkpoint at epoch $DONE_EPOCH >= EPOCHS=$EPOCHS — chain complete"
    break
  fi

  # Wall-clock guard: ensure >=30 min remains before launching another cycle
  ELAPSED=$((SECONDS - CHAIN_START_TIME))
  MAX_SECONDS=$((MAX_WALL_CLOCK_HOURS * 3600))
  if [ "$DONE_EPOCH" -gt 0 ] && [ $((MAX_SECONDS - ELAPSED)) -lt 1800 ]; then
    log "wall-clock limit approaching ($((ELAPSED / 3600))h elapsed of ${MAX_WALL_CLOCK_HOURS}h) — stopping chain cleanly at epoch $DONE_EPOCH"
    break
  fi

  RESUME_ARGS=()
  [ -f "$LAST_PT" ] && RESUME_ARGS=(--resume "$LAST_PT")
  NEXT_EPOCH=$((DONE_EPOCH + 1))
  CYCLE_START=$SECONDS
  log "training epoch $NEXT_EPOCH/$EPOCHS (checkpoint: $DONE_EPOCH, elapsed: $((ELAPSED / 60))m)"

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
        # The completed sentinel (-1) means the process is exiting on its
        # own (early stop or final epoch): let it finish the post-save tail
        # (plots, best-model copy) instead of killing it mid-way.
        DONE_MARKER=$(pixi run python -c "
import torch
print(torch.load('$LAST_PT', map_location='cpu', weights_only=False).get('epoch'))
" 2>/dev/null | tail -1)
        if [ "$DONE_MARKER" = "-1" ]; then
          log "checkpoint marks training complete — waiting for natural exit"
          wait "$WRAPPER"
          log "process exited on its own (rc=$?)"
          break
        fi
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
    CYCLE_SEC=$((SECONDS - CYCLE_START))
    log "epoch $NEXT_EPOCH saved in ${CYCLE_SEC}s (~$((CYCLE_SEC / 60))m); memory released by process exit"
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
