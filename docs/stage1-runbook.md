# Stage 1 Training Runbook

End-to-end procedure for the SAHI-parity Stage 1 YOLO model. Read this
top-to-bottom once, then use the **TL;DR** section as the quick reference.

## TL;DR

```bash
# 1. Convert each raw dataset → staging (4 staging dirs)
python scripts/unify/izutsumi.py
python scripts/unify/manga109.py
python scripts/unify/coco_heads.py --input-dir catnip-data/data/AnimeHeadsv3 --output-dir catnip-data/data/staging/ah_coco
python scripts/unify/yolo_heads.py              # ah_yolo (v1+v2+ani_face_detection)

# 2. Merge staging → unified + pre-slice for SAHI parity
python scripts/unify/stage1.py --slice --slice-workers 4

# 3. Train (imgsz=640 matches the slice size)
python scripts/train/stage1.py

# 4. Inference (SAHI uses the SAME slice params from config)
python -m catnip extract --input-dir data/manga --output-dir data/crops
```

Estimated end-to-end runtime on M3 Air: ~25 min of CPU work
(unify + slice) + GPU training time (Lightning Studio).

---

## Background: why we pre-slice

SAHI slices a full manga page into overlapping 640×640 patches at
**inference** time (`src/detection/sahi_extractor.py:42-53`). The
**training** pipeline used to feed YOLO the unsliced full pages at
`imgsz=1280`. That's a train/inference mismatch in two ways:

| Aspect             | Old training        | SAHI inference     |
|--------------------|---------------------|--------------------|
| Imgsz              | 1280                | 640                |
| Object pixel scale | 30×30 head in 1280² | 100×100 head in 640² |
| Context            | full page           | 640×640 crop       |

The new pipeline pre-slices the training images with the **exact same
SAHI parameters** (`params.sahi.slice_height/width`, `params.sahi.overlap_ratio`)
and trains YOLO at `imgsz=640`. Train and inference now see the same
distribution of object scales and contexts.

**Trade-offs:**
- Dataset grows by ~1.1–1.2× (only the ~100 manga pages meaningfully
  expand; the 80k+ head-crop images are already 640-ish in one dim and
  slice to 1 patch each).
- 300 epochs at 80k images may overfit; consider halving to 150 epochs
  on the first run and re-evaluating.
- Duplicate near-boundary boxes across overlapping slices. This is
  intentional — SAHI's NMS at inference mirrors the desired de-dup
  behaviour.

---

## Storage layout (Cloudflare R2, `s3://catnip-canonical`)

### Current → target

| Prefix                              | Status            | Notes |
|-------------------------------------|-------------------|-------|
| `data/manga/`                       | ✅ **KEEP**        | Source manga pages. Read-only to training. |
| `data/izutsumi/manga/`              | ✅ **KEEP**        | Izutsumi-annotated manga pages. |
| `data/izutsumi/annotations/`        | ✅ **KEEP**        | Per-annotation Label Studio JSONs. |
| `data/ls-exports/`                  | ✅ **KEEP**        | Raw Label Studio export bundles. |
| `data/manga109/`                    | ✅ **KEEP**        | manga109 XML + images (public dataset, but kept for traceability). |
| `data/anime_head_detection/`        | ✅ **KEEP**        | deepghs datasets (v1, v2, ani_face_detection). |
| `data/AnimeHeadsv3/`                | ✅ **KEEP**        | COCO-format dataset. |
| `data/staging/`                     | ⚠️ **REGENERATE**  | Output of `scripts/unify/*.py` convertors. Safe to delete after training. |
| `data/annotations/v01-v12/`         | 🗑 **ARCHIVE**     | Orphaned annotation versions. Archived during the GCS-era cleanup. |
| `data/annotations/latest_export.json` | 🆕 **CREATE**    | Canonical "latest" copy, refreshed on every Label Studio export. |
| `data/stage1/`                      | 🗑 **DELETE/ARCHIVE** | Legacy `setup_stage1_data` symlink-based output. Superseded by `training/stage1/`. |
| `training/stage1/`                  | 🆕 **CREATE**     | Unified YOLO dataset from `scripts/unify/stage1.py`. ~80k images, ~14 GB. |
| `training/stage1/dataset.yaml`      | 🆕 **CREATE**     | YOLO dataset manifest. |
| `training/stage1/split_manifest.json` | 🆕 **CREATE**   | Per-source-per-split counts. |
| `training/stage1_sliced/`           | 🆕 **CREATE**     | **Pre-sliced** training data (SAHI parity). ~80k images → ~80k–95k slices. ~14 GB. |
| `training/stage1_sliced/dataset.yaml` | 🆕 **CREATE**   | YOLO manifest for the sliced dataset. |
| `training/stage2/`                  | ⏸ **DEFER**       | Stage 2 ReID. Out of scope for Stage 1 run. |
| `models/yolo26n.pt`                 | ✅ **KEEP**        | Pretrained YOLO26n weights from Ultralytics (~10 MB). |
| `models/yolo26_stage1_body_head_face.pt` | 🆕 **CREATE** | **Trained Stage 1 model** (the deliverable). ~20 MB. |
| `models/best.pt`                    | 🗑 **DELETE**     | Renamed/copied to `yolo26_stage1_body_head_face.pt` after training. |
| `runs/detect/stage1/`               | ⚠️ **KEEP LAST**   | Ultralytics training output (plots, results.csv, weights/). Keep the most recent run only. |
| `runs/izutsumi*`                    | 🗑 **ARCHIVE**     | Old training runs. Archived during the GCS-era cleanup. |
| `runs/0.11.*`                       | 🗑 **ARCHIVE**     | Old run names. Archived during the GCS-era cleanup. |
| `results/`                          | 🗑 **ARCHIVE**     | Old inference results. Archived during the GCS-era cleanup. |
| `archive/`                          | ✅ **KEEP**        | GCS-era archive of stale runs/results. |

### Cleanup

The GCS-era cleanup (stale runs, orphaned annotation versions, `.DS_Store`
droppings) is done — `scripts/reorganize_gcs.py` was removed after it
served its purpose. R2 object CRUD uses `wrangler` (single objects) or
`s5cmd` against the S3 API (bulk). Credentials recipe: `.env.example`.

### What to upload / sync to R2

| Local path                         | Sync to GCS                            | When |
|------------------------------------|----------------------------------------|------|
| `catnip-data/data/ls-exports/`     | `s3://catnip-canonical/data/ls-exports/`    | After every Label Studio export. |
| `catnip-data/data/izutsumi/annotations/` | `s3://catnip-canonical/data/izutsumi/annotations/` | After every Label Studio export. |
| `catnip-data/training/stage1_sliced/` | `s3://catnip-canonical/training/stage1_sliced/` | After `scripts/unify/stage1.py --slice`. |
| `catnip-data/models/yolo26_stage1_body_head_face.pt` | `s3://catnip-canonical/models/` | After `scripts/train/stage1.py` completes. **Required**. |
| `catnip-data/runs/detect/stage1/` | `s3://catnip-canonical/runs/detect/stage1/` | Optional but useful (training curves, results.csv). |

Sync commands (R2 S3 API; map `R2_*` creds to `AWS_*` env, `AWS_REGION=auto`):

```bash
s5cmd --endpoint-url $R2_ENDPOINT sync catnip-data/data/ls-exports/ s3://catnip-canonical/data/ls-exports/
s5cmd --endpoint-url $R2_ENDPOINT sync catnip-data/training/stage1_sliced/ s3://catnip-canonical/training/stage1_sliced/
npx wrangler r2 object put catnip-canonical/models/yolo26_stage1_body_head_face.pt \
    --file catnip-data/models/yolo26_stage1_body_head_face.pt --remote
```

### What can be safely deleted locally

| Local path                         | Safe to delete? | Why |
|------------------------------------|-----------------|-----|
| `catnip-data/data/staging/`        | ✅ Yes          | Regenerated by `scripts/unify/*.py`. |
| `catnip-data/data/stage1/`         | ✅ Yes          | Legacy `setup_stage1_data` symlink tree. Superseded by `training/stage1/`. |
| `catnip-data/training/stage1/`     | ⚠️ After slicing | The sliced dataset derives from it but the slicer reads from it. Keep until `--slice` completes. |
| `catnip-data/training/stage1_sliced/` | ❌ No (deliverable) | **The training data**. |
| `catnip-data/runs/detect/stage1/`  | ⚠️ Keep latest only | Older run dirs can be deleted once the new model is in `models/`. |
| `catnip-data/models/yolo26n.pt`    | ❌ No (pretrain) | Needed to resume fine-tuning. |
| `catnip-data/models/yolo26_stage1_*.pt` | ❌ No (deliverable) | **The model**. |
| `catnip-data/data/izutsumi/`, `catnip-data/data/manga109/`, `catnip-data/data/AnimeHeadsv3/`, `catnip-data/data/anime_head_detection/` | ❌ No (source data) | Re-download is slow/expensive. |

---

## End-to-end run order

All commands assume CWD = repo root and pixi env active (`pixi shell`).

### A. On the dev machine (M3 Mac Air)

```bash
# 1. Generate staging directories (pixi tasks wrap the same scripts)
pixi run unify-izutsumi
pixi run unify-manga109
pixi run unify-coco-heads
pixi run unify-yolo-heads

# 2. Merge + slice for SAHI parity
pixi run unify-stage1              # = python scripts/unify/stage1.py --slice --slice-workers 4

# 3. (Optional) Sync sliced dataset to Kaggle (creates a new version of the
#    private Kaggle dataset `catnip-stage1-sliced`).
#    Requires KAGGLE_API_TOKEN in .env (from kaggle.com/settings/api).
pixi run -e kaggle kaggle-sync
```

### B. On Lightning AI Studio (T4/L4 GPU) — primary training path

R2 is the data source (zero egress). Full ingest + env + train +
checkpoint procedure: `docs/r2-migration-plan.md` Phase 4. Short version:

```bash
curl -fsSL https://pixi.sh/install.sh | sh   # pixi is not preinstalled
mkdir -p ~/catnip-data
npx wrangler r2 object get catnip-canonical/compressed/stage1_sliced.tar \
    --file ~/catnip-data/stage1_sliced.tar --remote
tar -xf ~/catnip-data/stage1_sliced.tar -C ~/catnip-data && rm ~/catnip-data/stage1_sliced.tar
export CATNIP_DATA=$HOME/catnip-data
pixi install -e cuda    # cuda env = conda pytorch 2.6.0 (CUDA 12.6) build
pixi run -e cuda python -c "import torch; print(torch.cuda.is_available())"   # True
```

Train (free-plan T4): `pixi run -e cuda train-stage1 --override training.stage1.device=cuda`

Push the model + run back to R2 after each chunk (see the migration plan
Phase 4 step 7 — `wrangler r2 object put ... --remote`).

Free-plan budget: ~15 credits/month ≈ 22 T4-hr; 4 h restart cycle
(`--resume` + R2 checkpoint tarballs mitigate).

### B2. On Kaggle (T4×2 GPU) — fallback

Once-per-account setup:

1. Verify phone number at kaggle.com/settings (required for GPU access).
2. Generate API token at kaggle.com/settings/api → "Generate New Token".
   Add it as a Kaggle Secret named `KAGGLE_API_TOKEN`
   (Add-ons → Secrets in the notebook sidebar).
3. Create the private dataset `catnip-stage1-sliced` via the Kaggle web
   UI (one-time), then run step A.4 to push the first version. In the
   notebook sidebar, attach the dataset under "Add data".

Per-session:

```python
# 1. Open notebooks/catnipKaggle.ipynb in a new notebook.
# 2. Right sidebar: Accelerator = GPU T4 x2, Internet = ON.
# 3. Add data: catnip-stage1-sliced (and, for resume, the most recent
#    catnip-stage1-output version).
# 4. Run all three cells.
#    - Cell 1: install deps, hydrate KAGGLE_API_TOKEN secret, clone repo,
#      copy the sliced dataset to /kaggle/working/, set CATNIP_DATA.
#    - Cell 2: trains 24 epochs at batch=64 (≈12 h); resumes from last.pt
#      if present in /kaggle/working/catnip-data-local/runs/detect/stage1/.
#    - Cell 3: publishes the model + run to a new `catnip-stage1-output`
#      version (private) via kagglehub. /kaggle/working/ is wiped at
#      session end — this is the only way to keep the artifact.

# To resume after the 12 h cap, re-attach the latest catnip-stage1-output
# as a second dataset, copy its weights/ into the canonical path, and
# re-run cells 2-3.
```

Quotas to budget around:

| Limit                        | Value          | Implication                                |
| ---------------------------- | -------------- | ------------------------------------------ |
| Weekly GPU hours             | 30 h           | ~2 long sessions per week                  |
| Per-session runtime          | 12 h           | Cap on a single training chunk (≈24 epochs) |
| Working disk per session     | 20 GB          | The 14 GB dataset + cache fits, barely     |
| Per-dataset storage          | 200 GB         | Sequential versions of `catnip-stage1-sliced` are fine |
| Private-datasets total       | 200 GB         | Two datasets × sequential versions is fine |

### B3. On a spot GPU VM (Vast.ai / RunPod) — recommended while Lightning is down

4090-class spot instances run the 150-epoch stage1 (halved from 300,
patience=15) in ~15–20 h at ~$0.20–0.45/h (≈ $3–9 total). Egress from R2
is free, so the 14 GB ingest costs nothing on any host. Interruption-safe:
checkpoints push back to R2 every `CKPT_INTERVAL_MIN` and a re-run
resumes from `s3://catnip-canonical/runs/stage1-last.pt`.

Pick any CUDA 12.x PyTorch image (Vast.ai: "PyTorch" templates; RunPod:
`runpod/pytorch:2.x-cuda12.x`). Then:

```bash
# 1. Provide the R2 token (R2_ACCESS_KEY_ID/R2_SECRET_ACCESS_KEY from
#    ~/secrets/r2-sippy.conf on the dev machine — export or write to a file
#    and pass R2_CREDS=<file>).
# 2. One command — clone, ingest from R2, train, publish back to R2:
bash scripts/train/spot_bootstrap.sh

# Tunables (env):
#   SPOT_BATCH=48 SPOT_WORKERS=16 CKPT_INTERVAL_MIN=10
#   CATNIP_REPO=~/catnip CATNIP_DATA=~/catnip-data
#   CATNIP_BRANCH=refactor/reID   (switch to main once merged)
```

What the script does (and assumes):

- Preflight: GPU visible (`nvidia-smi`), ≥40 GB free disk, R2 creds.
- Installs pixi + the `cuda` env (conda pytorch, CUDA 12.6) if missing.
- Pulls `compressed/stage1_sliced.tar` from R2 (s5cmd if installed,
  else boto3 multipart — a `~/.catnip-r2-venv` is built automatically when
  system python lacks boto3 or pip is PEP 668-restricted), extracts,
  **repoints `dataset.yaml` `path:` to the local root** (the tarball ships
  with the generating host's absolute path).
- Hydrates `runs/detect/stage1/weights/last.pt` from R2 if present, then
  `train-stage1 --resume` (ultralytics restores epoch/optimizer state).
  NOTE: ultralytics resume restores saved args from the checkpoint; the
  explicit `data=`/`batch` overrides win, but if a resume run complains about
  the old `/content/...` path, delete the stale `args.yaml` beside the
  checkpoint and start fresh.
- Watcher pushes `last.pt` to `runs/stage1-last.pt` every 10 min (a
  partially-written `last.pt` can occasionally make a push corrupt — harmless,
  the next push overwrites it; verify sizes if resuming from a suspect push).
- On completion: publishes `models/yolo26_stage1_body_head_face.pt` and a
  `runs/stage1-run.tar` to R2.

Budget guidance: a Vast.ai 4090 (~$0.20–0.30/h) is cheapest but hosts are
flakier — fine given the resume loop. RunPod community 4090 (~$0.34/h) is
steadier. Both fit well under a $25 cap.

### C. On Colab (T4 GPU) — fallback only

Last-resort fallback. Colab has no predictable quota; sessions can be
terminated without notice.

```python
# 1. Open notebooks/catnipColab.ipynb
# 2. Add R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY as Colab secrets
# 3. Set BRANCH in the setup cell (currently refactor/reID)
# 4. Run all cells — R2 tarball download (boto3), train, upload back
```

### D. Back on the dev machine

The full local flow (HDD/R2 data source, MPS training, results push to R2)
is notebook-driven: `notebooks/catnipLocal.ipynb`. Not the production path —
MPS is ~3-5x slower than a T4; use for smoke tests and debug runs. For real
(stratified-subset) local training, see section E below.

```bash
# 4. Pull the trained model from R2
npx wrangler r2 object get catnip-canonical/models/yolo26_stage1_body_head_face.pt \
    --file catnip-data/models/yolo26_stage1_body_head_face.pt --remote

# 5. (Optional) Run inference on a held-out page to sanity-check
python -m catnip extract \
    --input-dir catnip-data/data/manga/v09 \
    --output-dir /tmp/crops
ls /tmp/crops
```

---

### E. On the M3 Air itself — subset training (`exp/m3-local-stage1`, experimental)

The full ~165k-slice dataset at 150 epochs is ~80 MPS-days — but most of it
is redundancy: overlapping page slices (manga109 alone expanded 7.3x under
slicing) and near-duplicate single-object head crops. The experimental branch
trains on a stratified ~10% subset at the full 640px — SAHI parity intact,
imgsz untouched:

```bash
pixi run subset-stage1     # build training/stage1_sliced_subset (symlinks, ~1 min)
pixi run train-stage1-m3-chain   # 40 epochs, one epoch per process (~10-12 h overnight)
```

Measured on the M3 Air (16 GB, torch 2.4.1, ultralytics 8.4.193):

| Config                          | Speed            | GPU mem |
|---------------------------------|------------------|---------|
| batch 8, workers 0 (old clamp)  | 0.35 s/img       | 5.4 GB  |
| batch 16, workers 4              | **0.075 s/img**  | 4.3 GB  |
| batch 32, workers 4              | 0.106 s/img      | 8.2 GB  |
| batch 16, workers 8              | 0.10 s/img       | 4.7 GB  |

The old 2.8 s/it baseline was NOT GPU compute — ultralytics clamps
`workers=0` on MPS (`BaseTrainer.__init__`), so CPU-side augmentation ran
single-threaded and dominated. `scripts/train/stage1.py` now restores the
configured workers on MPS (`_patch_mps_workers`); verify the log says
"Using 4 dataloader workers".

**Memory caveat (16 GB machines) — the MPS driver-memory leak**:
ultralytics-on-MPS retains Metal driver memory ~3-5 MB per training
iteration (rate scales with dataset size; does NOT saturate, survives
`gc.collect()` + `torch.mps.empty_cache()`, invisible to the torch
allocator). Isolated to the ultralytics training loop: bare conv/BN/
upsample/H2D/AdamW loops on the same torch 2.6 + macOS 27 are flat, and
small-fraction ultralytics runs (<=1.4k images) barely show it — a full
subset epoch (~5.2k images, 652 iters) climbs ~2G/epoch and kills the
process by epoch 2-3 (jetsam "low-swap", exit 137). Also: torch 2.4.1
leaked the same, deterministic-mode and dataloader workers are
exonerated, and torchvision NMS needs `PYTORCH_ENABLE_MPS_FALLBACK=1`
(stage1.py sets it).

Consequences for the M3 run:
- the subset knobs are sized so ONE epoch fits: floor ~2G + ~3.5G
  retention + ~1.2G reclaimable peak ≈ 7G, below the practical ~8-9G
  jetsam line with normal apps open;
- multi-epoch single processes are impossible → `pixi run
  train-stage1-m3-chain` runs one epoch per process and resumes
  (`scripts/train/m3_chain.sh`), which restores epoch/optimizer/EMA and
  keeps the cos_lr schedule and early stopping exactly as one
  continuous 40-epoch run;
- `_install_mps_cache_guard` bounds the reclaimable peak mid-epoch and
  logs driver-vs-allocated memory every 50 batches;
- batch 8 is the default: throughput is augmentation-bound, so the
  smaller footprint is free.

First full run (2026-10-31, exp/m3-local-stage1): 40-epoch chain early-
stopped at epoch 24 (best epoch 12, patience 12). Full-test-split val of
`yolo26_stage1_body_head_face_m3.pt` (15,993 imgs / 87,544 boxes):
**mAP50 0.696, mAP50-95 0.374** (body 0.729/0.435, head 0.638/0.349,
face 0.719/0.340) — ~10.5h wall-clock on the M3 Air, zero jetsam events
after the chain was in place.

Subset design (knobs in `training.stage1.subset`, rebuild after changing):

- 13,539 train / 1,521 val slices; class ratios within a few % of full.
- izutsumi 100%, manga109 capped at 1 slice per page (every page kept),
  ah_yolo 8%, ah_coco 10% — head(1) exists only in ah_*, so no source hits 0.
- Artifacts stay separate: `runs/detect/stage1_m3*`, best model saved as
  `models/yolo26_stage1_body_head_face_m3.pt` — canonical names untouched.

Caveats:

- No fan: sustained load may throttle past the first hour; keep it plugged
  in with airflow. Expect the 12 h estimate to stretch, not shrink.
- 1.5k-image val → ~±1% mAP noise on early-stop and best-epoch selection.
- The honest final number comes from the full (never subset) test split:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 pixi run python -c "
from ultralytics import YOLO
from src.config import settings
YOLO(str(settings.paths.model_dir / 'yolo26_stage1_body_head_face_m3.pt')).val(
    data=str(settings.paths.stage1_sliced_dir / 'dataset.yaml'),
    split='test', imgsz=640, device='mps',
)"
```

---

## Verifying the slicer

Quick sanity check after step 3:

```python
from src.config import settings
from src.training.slicing import slice_split

# 1) Slice parameters must match training imgsz
assert settings.params.sahi.slice_height == settings.training.stage1.imgsz
assert settings.params.sahi.slice_width  == settings.training.stage1.imgsz

# 2) Negative samples must produce empty label files
from PIL import Image
import tempfile, os
with tempfile.TemporaryDirectory() as d:
    Image.new("RGB", (640, 640), "white").save(f"{d}/neg.jpg")
    open(f"{d}/neg.txt", "w").close()
    stats = slice_split(f"{d}", f"{d}", f"{d}/o_i", f"{d}/o_l",
                        slice_height=640, slice_width=640,
                        overlap_height_ratio=0.2, overlap_width_ratio=0.2,
                        min_area_ratio=0.3)
    assert os.path.getsize(f"{d}/o_l/neg_0_0_640_640.txt") == 0
    print("slicer OK:", stats)
```

---

## Failure modes

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| MPS run crawls (~2.8 s/it @ batch 8) | ultralytics clamps `workers=0` on MPS; CPU aug dominates | Fixed by `_patch_mps_workers` in `scripts/train/stage1.py`; log must say "Using 4 dataloader workers" |
| `s5cmd`/`wrangler` not found | R2 tooling missing | `brew install s5cmd`; `npx wrangler`. Map `R2_*` creds to `AWS_*` env vars. |
| Slicer hangs on first run | Cold PIL/SAHI import | Normal — first call is slow. |
| `AttributeError: 'Paths' object has no attribute 'stage1_sliced_dir'` | Old `Settings` pydantic model in checkpoint | `rm -rf .pixi/envs/default/lib/python3.11/site-packages/catnip*` (after `pixi install`). |
| `RuntimeError: invalid image size` | Empty/corrupt source image | Inspect the file; rerun skip the bad file. |
| mAP plateaus below 0.7 on tiny heads | imgsz drifted from slice size | Re-check `training.stage1.imgsz == params.sahi.slice_height`. |
| Colab OOM | Batch too large for T4 | The OOM retry in `scripts/train/stage1.py` drops to batch=4 then 2. |
| Val mAP much lower than train mAP | Train/val drift in slice parameters | Re-run `unify/stage1.py --slice` with the same `params.sahi.*`. |
| Kaggle: accelerators greyed out | Phone not verified or weekly quota hit | Verify phone at kaggle.com/settings; check kaggle.com/me/quota. |
| Kaggle: `No module named 'kaggle_secrets'` | Internet OFF in notebook session | Toggle Internet ON in the right sidebar and re-run cell 1. |
| Kaggle: `OSError: [Errno 28] No space left` at end of session | 20 GB cap reached by per-split `.cache` files | `kaggle_publish.py` strips caches automatically; if it still fails, raise `cache: false` in `model.train()` and re-run. |
| Kaggle: model disappeared after session | Forgot Cell 3 (or it failed) | Re-run Cell 3 against the most recent `catnip-stage1-output` version; the model is recoverable. |
| Kaggle: `401 Unauthorized` or `UnauthenticatedError` | API token expired or not set | Re-generate KAGGLE_API_TOKEN at kaggle.com/settings/api; verify the secret/env var is present. |
| Kaggle: dataset upload stalls on 14 GB | Network timeout on large dataset | Wait; if it stays stalled for >30 min, re-run `kaggle_sync.py` (kagglehub retries resume-ably). |
