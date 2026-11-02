# Stage 1 on the M3 Air — handover log (exp/m3-local-stage1)

Record of the 2026-10-30/31 experiment: train stage1 (YOLO26n face/head/body
detection) locally on an M3 MacBook Air (16 GB, 8 cores, no fan) in a
reasonable timeframe, keeping accuracy as high as possible. Everything below
was done on branch `exp/m3-local-stage1` (off `refactor/reID`); nothing was
pushed and no remote (R2, Lightning, Colab, Kaggle) was touched.

Machine: M3 Air, 16 GB unified memory, macOS 27.0.1, conda torch (2.4.1 at
start, 2.6.0 by the end), ultralytics 8.4.193, dataset on
`/Volumes/rifuSSD/catnip-data`.

## TL;DR — the outcome

- **Trained model delivered**: `models/yolo26_stage1_body_head_face_m3.pt`
  (best epoch 12 of an early-stopped 24-epoch run, ~10.5h wall-clock).
- **Honest metric** — full test split (15,993 images, 87,544 boxes, never
  subset): **mAP50 0.696, mAP50-95 0.374** (body 0.729/0.435, head 0.638/0.349,
  face 0.719/0.340).
- The path there required fixing three independent blockers; the third
  (an MPS driver-memory leak in the ultralytics training loop) cannot be
  fixed, only engineered around — the final architecture is a
  one-epoch-per-process resume chain (`scripts/train/m3_chain.sh`).

## The baseline problem

`pixi run train-stage1 --override training.stage1.device=mps` on the full
SAHI-sliced dataset (132,435 train / 16,264 val slices, 640px) measured
**2.8 s/it at batch 8 → ~12.9 h/epoch → ~80 GPU-days** for 150 epochs.
Unusable.

## Blocker 1 — ultralytics zeroes dataloader workers on MPS (fixed)

`ultralytics/engine/trainer.py` `BaseTrainer.__init__` hard-forces
`self.args.workers = 0` for cpu/mps ("faster CPU training as time dominated
by inference"). That assumption is inverted for yolo26n @640: the nano
model's compute is tiny and CPU-side augmentation (mosaic, HSV, decode)
dominates. The 2.8 s/it baseline was single-threaded dataloading, not GPU.

Fix: `_patch_mps_workers` in `scripts/train/stage1.py` — wraps
`BaseTrainer.__init__` and restores the configured workers on MPS only
(guarded against double-wrap; the OOM retry loop re-enters `model.train`).

Measured after the fix (workers 4, subset data, torch 2.6):

| Config               | Speed        | MPS driver mem | Verdict |
|----------------------|--------------|----------------|---------|
| batch 8, workers 0   | 0.35 s/img   | ~5.4 GB        | old baseline |
| batch 16, workers 4  | **0.075 s/img** | 4.3 GB      | fastest, but memory-lethal (see blocker 3) |
| batch 32, workers 4  | 0.106 s/img  | 8.2 GB         | worse |
| batch 16, workers 8  | 0.10 s/img   | 4.7 GB         | worse (E-core/spawn contention) |
| batch 8, workers 4   | ~0.09 s/img  | ~4 GB peak     | **production choice** |

Throughput is augmentation-bound: 4 workers feed ~13 img/s regardless of
batch size, so batch 8's smaller footprint is free wall-clock.

## Blocker 2 — torchvision NMS is unimplemented on MPS (fixed)

Every epoch-end validation crashed with
`NotImplementedError: The operator 'torchvision::nms' is not currently
implemented for the MPS device`. Fix: `PYTORCH_ENABLE_MPS_FALLBACK=1` —
set via `os.environ.setdefault` before the torch import in
`scripts/train/stage1.py`. Ad-hoc val commands must prefix it themselves
(the runbook §E snippet includes it).

## Blocker 3 — MPS driver-memory leak in the ultralytics loop (worked around)

**Symptom**: macOS jetsam killed training processes ("low-swap" reason,
exit 137) mid-epoch. Driver-allocated memory grew monotonically: 4.3 → 8.2 GB
over ~650 iterations at batch 16.

**Characterization** (the numbers that matter):

- ~3-5 MB Metal driver memory retained per training *iteration*; rate
  scales with dataset size (5.3 MB/it on 13.5k images, 3.1 MB/it on 5.2k,
  ~2.4 MB/it on 1.4k) and never saturates — an epoch-2 rerun of the same
  data climbs as fast as epoch 1.
- Invisible to torch's allocator: `torch.mps.current_allocated_memory()`
  stayed flat at ~0.14-0.19 GB while `driver_allocated_memory()` climbed.
- Unreclaimable: `gc.collect()` + `torch.mps.empty_cache()` recover a
  constant ~1.1 GB chunk but the post-clear floor keeps rising.
- Survives torch 2.4.1 → 2.6.0.
- A 2-epoch canary on the 5.2k subset survived at a 9.8 GB floor but its
  epoch-2 validation stalled in swap thrash. A 1-epoch process peaks at
  ~5.2 GB. Practical process budget: **~650-1300 iterations**.

**Exonerated by isolation probes** (all FLAT in 600-iteration loops):
bare conv net fwd/bwd/AdamW, +BatchNorm/SiLU, +Upsample(nearest),
+softmax/topk/gather (loss-head style ops), per-iteration fresh-tensor
H2D with `non_blocking=True`, `driver_allocated_memory()` query spam
(300k calls), deterministic-mode on/off, dataloader workers 0/4/8,
batch 8/16/32. The leak reproduces **only inside the ultralytics trainer
loop**, and only at meaningful dataset scale (fraction 0.02 → flat,
0.10 → flat after warm-up, 1.0 → leak). Suspected mechanism: per-shape
kernel/specialization retention in the MPS backend (loss-path tensors
vary with per-batch gt-box composition), but this was never pinned to a
specific op. The core probe, for whoever picks this up:

```python
# flat = torch/MPS itself is fine
model = nn.Sequential(nn.Conv2d(3,32,3,2,1), nn.Conv2d(32,64,3,2,1), ...).to("mps")
x = cpu_batch.to("mps", non_blocking=True); loss = mse(model(x), t); backward; step()
# → driver_allocated_memory() stays flat over 600 iters
# same loop via YOLO(...).train(...) on ≥5k images → climbs ~3-5 MB/it
```

**The workaround** (what the final run uses):

1. Memory-safe subset: 5,214 train / 622 val slices (knobs in
   `config/pipeline.yaml` → `training.stage1.subset`; build with
   `pixi run subset-stage1`, deterministic seed, symlink-based,
   wipe-guarded). Sized so one epoch fits: ~2 GB floor + ~3.5 GB one-time
   retention + ~1.2 GB reclaimable peak ≈ 5.2 GB, below the practical
   ~8-9 GB jetsam line with normal apps open.
2. `scripts/train/m3_chain.sh` (`pixi run train-stage1-m3-chain`): run ONE
   epoch per process, wait for the `last.pt` save, kill, resume the next
   epoch in a fresh process. Ultralytics resume restores
   epoch/optimizer/EMA, so the cos_lr schedule, early stopping, and
   metrics behave exactly as one continuous run. Chain is
   checkpoint-driven (handles fresh starts and interrupted chains).

Two chain bugs found and fixed during the night (both instructive):

- `check_resume` restores **all** args from the checkpoint — `epochs` is
  not overridable on resume. A seed checkpoint saved with `epochs=2` made
  every cycle believe it was the final epoch, saved with the `epoch=-1`
  "completed" sentinel, and the chain looped. Fix: the chain reads both
  the stored epoch and the saved `epochs` arg; `-1` breaks as complete, a
  mismatch aborts with instructions.
- The chain killed the process 10s after any `last.pt` save — including
  the final save of an early-stopping run, which cut off the post-save
  tail (plots, best-model copy). The model had to be copied from
  `best.pt` by hand. Fix: after a save, re-read the stored epoch; `-1`
  means wait for the process to exit on its own.

## What didn't work (dead ends, for the record)

- torch 2.4.1 → 2.6.0 upgrade: no change to the leak (kept anyway — it's
  the same pairing the CUDA feature uses, and `torch.load` gained the
  `weights_only` default flip which the chain handles).
- `deterministic: false` plumbed and tested: no effect on the leak (knob
  kept; both values documented).
- `torch.mps.empty_cache()` mid-epoch (every 50 batches, with gc): bounds
  the reclaimable peak but cannot touch the leak floor. Kept as
  observability instrumentation (`mps guard` log lines).
- Larger batch (32) and more workers (8): both measured slower.
- Single-process multi-epoch runs at any subset size ≥ ~6k: die by
  epoch 2-3.
- Training at reduced imgsz (e.g. 480/320): rejected by design — imgsz
  must stay 640 to preserve SAHI train/inference parity. The redundancy
  was cut from the *dataset*, not the pixels.

## Results

Training run (2026-10-30 22:00 → 10-31 03:49 local, ~10.5h, 24 chained
processes, zero kills after the fixes landed):

- Early stop at epoch 24/40 (best epoch 12, patience 12).
- Per-epoch val on the 622-image subset peaked at mAP50 0.637 /
  mAP50-95 0.346 (epoch 12), plateaued ~0.58-0.60 after.
- Full-test-split validation of the best model (the honest number):

```
class   Images  Instances  P      R      mAP50  mAP50-95
all     15993   87544      0.731  0.651  0.696  0.374
body    8095    35345      0.696  0.724  0.729  0.435
head    7395    27414      0.701  0.602  0.638  0.349
face    7546    24785      0.795  0.626  0.719  0.340
```

The full-test number beat the small-val estimate — the 622-image val
subset is pessimistic/noisy (±1% mAP on early-stop decisions).

## Artifacts and how to rerun

```
# Rebuild the memory-safe subset (deterministic; ~1 min)
pixi run subset-stage1

# The real local training path (one epoch per process, ~15 min/cycle)
pixi run train-stage1-m3-chain            # EPOCHS=40 default

# Single-process task — WILL die after ~1-2 epochs; smoke tests only
pixi run train-stage1-m3

# Honest final metric (full test split, never subset)
PYTORCH_ENABLE_MPS_FALLBACK=1 pixi run python -c "
from ultralytics import YOLO
from src.config import settings
YOLO(str(settings.paths.model_dir / 'yolo26_stage1_body_head_face_m3.pt')).val(
    data=str(settings.paths.stage1_sliced_dir / 'dataset.yaml'),
    split='test', imgsz=640, device='mps')
"
```

Key paths: subset at `training/stage1_sliced_subset/` (+ `subset_manifest.json`
with per-source counts), run at `runs/detect/stage1_m3/` (`results.csv`
appends across chain cycles), chain log at `runs/detect/chain.log`. The m3
model name (`*_m3.pt`) can never clobber the canonical
`yolo26_stage1_body_head_face.pt`.

## Open questions / possible next steps

- **Leak root cause**: not pinned to an op. Worth a torch-nightly test and
  a torch/MPSGraph issue with the repro above. Any fix there would let the
  chain collapse back to a single process and the subset grow back toward
  13.5k.
- **Accuracy headroom**: the subset knobs are the dial. manga109 at 0.32
  kept 32% of pages once each; ah_yolo at 0.035. More of either = better
  model, but each image adds ~0.3-0.7 MB one-time retention and the
  per-epoch time scales linearly.
- **mAP50-95 gap** (0.374 vs mAP50 0.696): localization quality is the
  weak axis — more manga109 slices (richer box shapes per page) would
  likely help more than volume of head crops.
- Stage 2 (reID) untouched, as scoped.
- Commits on `exp/m3-local-stage1`: 8dedfd1 → 3590aa7 (7 commits). Not
  merged anywhere; merge decision and the CUDA-vs-local question are open.
