# Stage 1 Training Runbook

End-to-end procedure for the SAHI-parity Stage 1 YOLO model. Read this
top-to-bottom once, then use the **TL;DR** section as the quick reference.

## TL;DR

```bash
# 1. Clean + restructure GCS bucket
python scripts/reorganize_gcs.py                # archive (default) or --delete
gsutil -m cp catnip-data/data/ls-exports/*.json gs://catnip-data/data/annotations/latest_export.json

# 2. Convert each raw dataset → staging (4 staging dirs)
python scripts/unify/izutsumi.py
python scripts/unify/manga109.py
python scripts/unify/coco_heads.py --input-dir catnip-data/data/AnimeHeadsv3 --output-dir catnip-data/data/staging/ah_coco
python scripts/unify/yolo_heads.py              # ah_yolo (v1+v2+ani_face_detection)

# 3. Merge staging → unified + pre-slice for SAHI parity
python scripts/unify/stage1.py --slice --slice-workers 4

# 4. Train (imgsz=640 matches the slice size)
python scripts/train/stage1.py

# 5. Inference (SAHI uses the SAME slice params from config)
python -m catnip extract --input-dir data/manga --output-dir data/crops
```

Estimated end-to-end runtime on M3 Air: ~25 min of CPU work
(unify + slice) + Colab T4 training time (~1 h).

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

## GCS bucket layout (`gs://catnip-data`)

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
| `data/annotations/v01-v12/`         | 🗑 **ARCHIVE**     | Orphaned annotation versions. `scripts/reorganize_gcs.py` handles this. |
| `data/annotations/latest_export.json` | 🆕 **CREATE**    | Canonical "latest" copy, refreshed by `reorganize_gcs.py`. |
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
| `runs/izutsumi*`                    | 🗑 **ARCHIVE**     | Old training runs. Handled by `scripts/reorganize_gcs.py`. |
| `runs/0.11.*`                       | 🗑 **ARCHIVE**     | Old run names. Handled by `scripts/reorganize_gcs.py`. |
| `results/`                          | 🗑 **ARCHIVE**     | Old inference results. Handled by `scripts/reorganize_gcs.py`. |
| `archive/`                          | 🆕 **CREATE**     | Destination for `scripts/reorganize_gcs.py` archive operations. |

### Cleanup script

`scripts/reorganize_gcs.py` is the single source of truth for bucket
cleanup. It:

1. Archives (or `--delete`) the four stale path globs:
   `runs/izutsumi*`, `runs/0.11.*`, `runs/detect/`, `results/`.
2. Creates the `data/stage1/{images,labels}/` directory prefixes (no
   longer needed for the new pipeline, but kept for backward compat).
3. Copies the newest Label Studio export to
   `data/annotations/latest_export.json`.

```bash
python scripts/reorganize_gcs.py --dry-run     # preview
python scripts/reorganize_gcs.py --force       # archive (default, mv to archive/)
python scripts/reorganize_gcs.py --delete --force   # permanent delete
```

The script refuses to touch the protected prefixes `data/manga/` and
`data/annotations/` (no overlap with the stale globs by design).

### What to upload / sync to GCS

| Local path                         | Sync to GCS                            | When |
|------------------------------------|----------------------------------------|------|
| `catnip-data/data/ls-exports/`     | `gs://catnip-data/data/ls-exports/`    | After every Label Studio export. |
| `catnip-data/data/izutsumi/annotations/` | `gs://catnip-data/data/izutsumi/annotations/` | After every Label Studio export. |
| `catnip-data/training/stage1_sliced/` | `gs://catnip-data/training/stage1_sliced/` | After `scripts/unify/stage1.py --slice`. **Required** — Colab reads from here. |
| `catnip-data/models/yolo26_stage1_body_head_face.pt` | `gs://catnip-data/models/` | After `scripts/train/stage1.py` completes. **Required**. |
| `catnip-data/runs/detect/stage1/` | `gs://catnip-data/runs/detect/stage1/` | Optional but useful (training curves, results.csv). |

Sync commands:

```bash
gsutil -m rsync -r catnip-data/data/ls-exports/    gs://catnip-data/data/ls-exports/
gsutil -m rsync -r catnip-data/training/stage1_sliced/ gs://catnip-data/training/stage1_sliced/
gsutil -m cp catnip-data/models/yolo26_stage1_body_head_face.pt gs://catnip-data/models/
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
# 1. Reorganize the bucket (once)
python scripts/reorganize_gcs.py --force

# 2. Generate staging directories (pixi tasks wrap the same scripts)
pixi run unify-izutsumi
pixi run unify-manga109
pixi run unify-coco-heads
pixi run unify-yolo-heads

# 3. Merge + slice for SAHI parity
pixi run unify-stage1              # = python scripts/unify/stage1.py --slice --slice-workers 4

# 4. Sync sliced dataset to Kaggle (creates a new version of the private
#    Kaggle dataset `catnip-stage1-sliced`; GCS sync is separate below).
#    Requires KAGGLE_API_TOKEN in .env (from kaggle.com/settings/api).
pixi run -e kaggle kaggle-sync

# 5. Sync the pretrained YOLO26n to GCS (one-time, if not already there)
gsutil cp catnip-data/models/yolo26n.pt gs://catnip-data/models/yolo26n.pt
```

### B. On Kaggle (T4×2 GPU) — primary training path

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

### C. On Colab (T4 GPU) — fallback only

Use this only if Kaggle quota is exhausted mid-Phase-B.  Colab has no
predictable quota; sessions can be terminated without notice.

```python
# 1. Open notebooks/catnipColab.ipynb
# 2. Set BRANCH in cell 1 (currently refactor/reID)
# 3. Run all cells — gcsfuse mount, tarball download, train, upload back
```

### D. Back on the dev machine

```bash
# 6. Pull the trained model from the Kaggle output dataset
python -c "
import kagglehub
path = kagglehub.dataset_download('rifusaki/catnip-stage1-output')
print(path)
"
# Then copy the model from the downloaded path to local/GCS
cp <downloaded-path>/models/yolo26_stage1_body_head_face.pt catnip-data/models/
gsutil cp catnip-data/models/yolo26_stage1_body_head_face.pt \
    gs://catnip-data/models/

# 7. (Optional) Run inference on a held-out page to sanity-check
python -m catnip extract \
    --input-dir catnip-data/data/manga/v09 \
    --output-dir /tmp/crops
ls /tmp/crops
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
| `gsutil` not found | Google Cloud SDK not installed | Install from cloud.google.com/sdk. |
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
