# Catnip: Agent Operational Guide

> **Audience:** AI Coding Agents (Gemini, Claude, Mistral, Codex, OpenCode).
> **Purpose:** Single source of operational truth, hardware gotchas, storage layouts, and active workflows. Read before modifying code or running tasks.

---

## 1. Project Overview & Architecture

Catnip is a two-stage manga character detection and re-identification pipeline designed to detect and identify **Izutsumi** (from *Dungeon Meshi*).

- **Stage 1 (Localization):** Slicing Aided Hyper Inference (SAHI) + fine-tuned **YOLO26 Nano** (`yolo26n`).
  - Target classes (3): `0: body`, `1: head`, `2: face`.
  - Input images are pre-sliced into 640×640 overlapping windows (`overlap_ratio=0.2`) to achieve byte-for-byte scale parity between training and inference.
- **Stage 2 (Re-Identification):** ResNet18 + GeM (Generalized Mean) Pooling + Triplet Margin Loss.
  - Target classes (2): `0: izutsumi`, `1: not_izutsumi`.
  - Produces 128-d L2-normalized embeddings matched via FAISS cosine similarity (`IndexFlatIP`).

---

## 2. Reality Check: Codebase Status

| Component | File Path | Reality |
|---|---|---|
| Config | `config/pipeline.yaml`, `src/config.py` | ✅ **Fully working.** Pydantic v2 + OmegaConf with dot-notation overrides. |
| Data Unification | `scripts/unify/*.py`, `src/preprocess/` | ✅ **Fully working.** Ingests Label Studio, Manga109, deepghs, and AnimeHeadsv3. |
| SAHI Slicer | `src/training/slicing.py` | ✅ **Fully working.** Pre-slices training sets at 640px. |
| Stage 1 Training | `scripts/train/stage1.py` | ✅ **Fully working.** MuSGD, MPS worker restoration, cache clearing, OOM fallback. |
| M3 Resume Chain | `scripts/train/m3_chain.sh` | ✅ **Fully working.** One-epoch-per-process runner to bypass the MPS memory leak. |
| Colab Headless Runner | `scripts/colab/launch_360.sh` | ✅ **Fully working.** 24/7 background runner driven by home server (`360 WSL`). |
| Spot VM Bootstrap | `scripts/train/spot_bootstrap.sh` | ✅ **Fully working.** Auto-bootstrap for Vast.ai / RunPod RTX 4090 instances. |
| SAHI Inference | `src/detection/sahi_extractor.py` | ✅ **Working.** SAHI wrapper using Non-Maximum Merging (NMM). |
| Stage 1 Deliverable | `models/yolo26_stage1_body_head_face_m3.pt` | ✅ **Trained.** Achieved mAP50 0.696 on 16k full test split. |
| Stage 2 Training | `scripts/train/stage2.py` | ⚠️ **Scaffold.** Prepares triplet dataset; PyTorch triplet training loop is pending. |
| ReID Modules | `src/reid/` | ⚠️ **Scaffold.** Model architecture, sampler, and FAISS indexer need implementation. |
| CLI Interface | `src/cli.py` | ⚠️ **Stubbed.** Subcommands `extract`, `embed`, `match` have placeholder error exits. |

---

## 3. The Hall of Gotchas (Hardware & Framework Quirks)

### Gotcha 1: The Apple Silicon (MPS) Driver Memory Leak
* **Symptom:** Training YOLO on MPS leaks ~3–5 MB of uncollectable Metal driver memory per iteration. It survives `gc.collect()` and `torch.mps.empty_cache()`. After ~1.5–2 epochs on a large dataset, unified memory saturates and macOS jetsam kills the process (exit code 137).
* **Fix:** **Never run multi-epoch single processes on MPS.** Always use `pixi run train-stage1-m3-chain` (calls `scripts/train/m3_chain.sh`), which trains exactly 1 epoch, saves `last.pt`, cleanly terminates, and resumes in a fresh process.

### Gotcha 2: Ultralytics Clamps Dataloader Workers to 0 on MPS
* **Symptom:** Training on MPS crawls at ~2.8 s/it because Ultralytics hardcodes `workers = 0` on macOS/CPU in `BaseTrainer.__init__`. Because YOLO26n is tiny, CPU-bound image decoding and mosaic augmentations choke the entire pipeline.
* **Fix:** `scripts/train/stage1.py` applies `_patch_mps_workers` to monkey-patch `BaseTrainer.__init__` and restore 4 workers on MPS (boosting throughput ~4.7× to 0.075 s/img).

### Gotcha 3: Torchvision NMS on MPS
* **Symptom:** Epoch-end validation crashes with `NotImplementedError: The operator 'torchvision::nms' is not currently implemented for the MPS device`.
* **Fix:** Always ensure `PYTORCH_ENABLE_MPS_FALLBACK=1` is set in the environment. `scripts/train/stage1.py` exports this on import.

### Gotcha 4: GCS is Decommissioned
* **Rule:** Google Cloud Storage (`gs://catnip-data`) has been fully decommissioned. **Never introduce or run `gsutil`, `gcsfuse`, or GCS API calls.** All remote storage is on Cloudflare R2 (`catnip-canonical`).

### Gotcha 5: CUDA Environment on Linux/VMs
* **Rule:** On Linux, Colab, or Spot instances, use `pixi run -e cuda ...`. The conda build string pin (`build = "*cuda*"`) in `pixi.toml` ensures compatibility with CUDA 12.6 without pip wheel collisions.

---

## 4. Storage & Data Conventions

- **Canonical Archive:** Cloudflare R2 bucket `catnip-canonical` (`s3://catnip-canonical`) with $0 egress.
- **Local Data Dir:** `catnip-data/` (or `/Volumes/rifuSSD/catnip-data`).
- **Tarball for Cloud Instances:** `compressed/stage1_sliced.tar` (~14 GB) contains the pre-sliced training dataset for instant hydration on remote VMs.
- **Rolling Checkpoints:** Remote runners push `runs/detect/stage1/weights/last.pt` to `s3://catnip-canonical/runs/stage1-last.pt` every 10 minutes. Training resumes seamlessly if interrupted.

---

## 5. Standard Pixi Tasks & Recipes

Run all commands from the repository root with Pixi:

```bash
# Data Preparation
pixi run data-prep             # Runs unify for izutsumi, manga109, coco, yolo heads + SAHI slicing
pixi run sanitize-names        # Normalizes filenames (removes quotes, spaces, unicode)

# Training Workflows
pixi run subset-stage1         # Builds stratified 10% subset symlinks for M3 training
pixi run train-stage1-m3-chain # Runs leak-safe 40-epoch chained training on M3 MPS
bash scripts/train/spot_bootstrap.sh  # Complete bootstrap + training on Vast.ai/RunPod

# Test Evaluation (MPS snippet)
PYTORCH_ENABLE_MPS_FALLBACK=1 pixi run python -c "
from ultralytics import YOLO
from src.config import settings
model = YOLO(str(settings.paths.model_dir / 'yolo26_stage1_body_head_face_m3.pt'))
model.val(data=str(settings.paths.stage1_sliced_dir / 'dataset.yaml'), split='test', imgsz=640, device='mps')
"
```

---

## 6. Baseline Metrics (Stage 1 Deliverable)

Delivered checkpoint: `models/yolo26_stage1_body_head_face_m3.pt` (trained on stratified subset via M3 chain):

| Split | Images | Boxes | Class | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| **Full Test Split** | **15,993** | **87,544** | **All** | **0.696** | **0.374** |
| | | | Body (0) | 0.729 | 0.435 |
| | | | Head (1) | 0.638 | 0.349 |
| | | | Face (2) | 0.719 | 0.340 |
