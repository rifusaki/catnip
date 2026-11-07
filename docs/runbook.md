# Catnip Operations Runbook

End-to-end operational guide for data preparation, SAHI pre-slicing, Stage 1 YOLO training, and checkpoint management.

> **Next-Gen Roadmap & Handoff:** See [`docs/implementation_plan.md`](implementation_plan.md) for the Stage 1 v2 (`yolo26m` L4 upgrade, cross-dataset pseudo-labeling, WBF, dynamic panel cropping) and Stage 2 Semi-Supervised Clustering implementation plan.

## TL;DR

```bash
# 1. Unify raw datasets + pre-slice for SAHI scale parity (all-in-one task)
pixi run data-prep

# 2. Sanitize filenames (strips quotes, spaces, non-ASCII for cloud safety)
pixi run sanitize-names

# 3. Train Stage 1 (Choose your compute):
# Option A: 24/7 Headless Colab on 360 WSL (Recommended)
ssh 360 "wsl tmux new -s catnip-training 'cd ~/catnip-colab && ./launch_360.sh'"

# Option B: Spot GPU VM (Vast.ai / RunPod RTX 4090)
bash scripts/train/spot_bootstrap.sh

# Option C: Local M3 Mac Air (Stratified subset, one epoch per process)
pixi run subset-stage1
pixi run train-stage1-m3-chain

# 4. Evaluate on full test split (MPS or CUDA)
PYTORCH_ENABLE_MPS_FALLBACK=1 pixi run python -c "
from ultralytics import YOLO
from src.config import settings
model_path = settings.paths.model_dir / 'yolo26_stage1_body_head_face_m3.pt'
YOLO(str(model_path)).val(
    data=str(settings.paths.stage1_sliced_dir / 'dataset.yaml'),
    split='test', imgsz=640, device='mps',
)
"

# 5. Comparative Evaluation against Anime Face & Head Benchmarks
pixi run benchmark-faces --sample 200        # Catnip vs Fuyucchi, zymk9, and DeepGHS
pixi run benchmark-heads --sample 200        # Catnip vs DeepGHS Head Nano
pixi run benchmark-faces --download-only     # Pre-download and cache catalog weights
```

---

## 1. Background: SAHI Pre-Slicing & Scale Parity

SAHI slices full manga pages into overlapping 640×640 patches at **inference** time (`src/detection/sahi_extractor.py`). Training directly on unsliced full pages at `imgsz=1280` produces a severe scale mismatch:

| Aspect | Unsliced Training | SAHI Sliced Inference | Pre-Sliced Training (Current) |
|---|---|---|---|
| Image size (`imgsz`) | 1280 | 640 | **640** |
| Tiny head pixel scale | ~30×30 in 1280² | ~100×100 in 640² | **~100×100 in 640²** |
| Context window | Full page | 640×640 crop | **640×640 crop** |

The unification pipeline pre-slices training images with the **exact same SAHI parameters** (`params.sahi.slice_height/width=640`, `params.sahi.overlap_ratio=0.2`) into `training/stage1_sliced/`. YOLO is trained at `imgsz=640` so the training distribution is byte-identical to SAHI window patches.

---

## 2. Data Preparation Pipeline

The raw inputs comprise four main collections:
1. `data/izutsumi`: Label Studio manga annotations (body, head, face).
2. `data/manga109`: 109 manga titles (XML annotations for body and face).
3. `data/anime_head_detection`: deepghs v1, v2, and ani_face_detection (head).
4. `data/AnimeHeadsv3`: COCO format anime heads (head).

### Running Unification

Run the full pipeline using the composite Pixi task:
```bash
pixi run data-prep
```
This automatically runs each step in order:
1. `unify-izutsumi`: Converts Label Studio JSONs to YOLO format.
2. `unify-manga109`: Normalizes Manga109 XML bounding boxes.
3. `unify-coco-heads`: Converts AnimeHeadsv3 COCO annotations.
4. `unify-yolo-heads`: Remaps deepghs head annotations to class `1`.
5. `unify-stage1`: Merges all staging directories into an 80/10/10 split in `training/stage1/`, then slices them with SAHI parameters into `training/stage1_sliced/`.

### Filename Sanitization
Some source datasets contain spaces, quotes, and non-ASCII characters that break remote uploaders or cloud tooling. Run:
```bash
pixi run sanitize-names
```

---

## 3. Storage Layout (Cloudflare R2, `s3://catnip-canonical`)

Cloudflare R2 (`catnip-canonical`) is the canonical cloud archive ($0 egress). Local SSD `/Volumes/rifuSSD/catnip-data` (or `~/catnip-data`) is the local working directory.

| Prefix | Status | Description |
|---|---|---|
| `data/manga/` | Read-only | Source manga page images. |
| `data/izutsumi/` | Read-only | Izutsumi-annotated manga pages & Label Studio JSONs. |
| `data/manga109/` | Read-only | Raw Manga109 dataset. |
| `data/anime_head_detection/` | Read-only | deepghs YOLO head datasets. |
| `data/AnimeHeadsv3/` | Read-only | COCO anime heads dataset. |
| `data/staging/` | Ephemeral | Intermediate YOLO conversion folders (`izutsumi`, `manga109`, `ah_yolo`, `ah_coco`). |
| `training/stage1/` | Derived | Unified unsliced dataset (~80k images). |
| `training/stage1_sliced/` | **Canonical** | **Pre-sliced training dataset (~165k slices, 14 GB)**. |
| `compressed/stage1_sliced.tar` | **Canonical** | Tarball of the sliced dataset for rapid VM/Colab hydration. |
| `models/yolo26n.pt` | Pretrained | Base Ultralytics YOLO26 nano weights. |
| `models/yolo26_stage1_body_head_face_m3.pt` | **Deliverable** | M3 local trained model (mAP50 0.696, mAP50-95 0.374). |
| `models/yolo26_stage1_body_head_face.pt` | **Deliverable** | Full run trained deliverable. |
| `runs/stage1-last.pt` | Checkpoint | Rolling 10-minute training checkpoint synced to R2 for resume. |

### R2 Sync Commands
Using `s5cmd` (via AWS S3 API compatibility) or `wrangler`:
```bash
# Push model to R2
npx wrangler r2 object put catnip-canonical/models/yolo26_stage1_body_head_face.pt \
    --file catnip-data/models/yolo26_stage1_body_head_face.pt --remote

# Pull model from R2
npx wrangler r2 object get catnip-canonical/models/yolo26_stage1_body_head_face.pt \
    --file catnip-data/models/yolo26_stage1_body_head_face.pt --remote
```

---

## 4. Compute & Training Workflows

### Option A: 24/7 Headless Colab on `360 WSL` (Recommended Remote)
Google Colab provides free/pro L4 or T4 GPUs, but browser disconnects kill long runs. The headless runner on `360 WSL` drives the Colab session via API, syncs checkpoints to R2 every 10 minutes, and automatically resumes from `runs/stage1-last.pt` if a disconnect occurs.

See `scripts/colab/README.md` for complete details:
```bash
# On 360 WSL:
cd ~/catnip-colab
tmux new -s catnip-training
./launch_360.sh
```

### Option B: Spot GPU VM (Vast.ai / RunPod RTX 4090)
Spot instances on Vast.ai or RunPod offer RTX 4090s at ~$0.20–0.40/h.
The automated bootstrap script handles R2 ingest, pixi cuda setup, training, and 10-minute checkpoint pushes:
```bash
# On the spot VM:
export R2_ACCESS_KEY_ID="<your_key>"
export R2_SECRET_ACCESS_KEY="<your_secret>"
bash scripts/train/spot_bootstrap.sh
```

### Option C: Local M3 Mac Air (Stratified Subset Chain)
Training the full 165k-slice dataset on Apple Silicon (MPS) is intractable (~80 GPU-days). Instead, we train on a stratified 10% subset (13,539 train / 1,521 val) at the full 640px resolution:
- Keeps 100% of Izutsumi annotations.
- Caps Manga109 to 1 slice per page.
- Retains 8% of `ah_yolo` and 10% of `ah_coco`.

**The MPS Metal Driver Leak:**
Ultralytics on MPS leaks ~3–5 MB of Metal driver memory per iteration, surviving `empty_cache()` and killing processes via macOS jetsam around epoch 2–3.
**Workaround:** We run a one-epoch-per-process chain (`scripts/train/m3_chain.sh`). Each epoch saves `last.pt`, the process terminates, and resumes fresh.

```bash
# Build symlink subset (~1 min)
pixi run subset-stage1

# Launch 40-epoch overnight chain (~10-12 h)
pixi run train-stage1-m3-chain
```

### Paused Training Paths
- **Lightning AI Studio:** Paused due to extended infrastructure outage.
- **Kaggle:** Paused because the dataset was rejected by Kaggle's automated NSFW scanner.

---

## 5. Comparative Model Evaluation & Benchmarks

The benchmark framework (`scripts/eval/benchmark_faces.py`, `src/eval/`) evaluates Catnip Stage 1 against prominent open-source anime face and head detectors:
- **`Fuyucch1/yolov8_animeface`** (`yolov8x6` @ 1280px, ~97M params, Safebooru dataset)
- **`zymk9/yolov5_anime`** (`yolov5s` @ 640px, ~7.2M params, Pixiv dataset)
- **`deepghs/anime_face_detection`** (`face_detect_v1.4_n` @ 640px, ~3.0M params — direct **Nano peer**)
- **`deepghs/anime_head_detection`** (`head_detect_v2.0_n_yv11` @ 640px, ~2.6M params — **Head peer**)

### Methodology & Alignment
1. **Class Remapping:** Isolates Catnip's 3-class system (`0: body`, `1: head`, `2: face`) and remaps target detections (`face` $\rightarrow$ 0, `head` $\rightarrow$ 0) for strict 1:1 evaluation against single-class external detectors.
2. **COCO Metric Standards:** Computes standard $\text{mAP}_{50}$ and $\text{mAP}_{50-95}$ using greedy matching over unmatched ground truth boxes.
3. **Domain Stratification:** Reports separate metrics for:
   - **Manga Line-Art** (`manga109`, `izutsumi`): evaluates adaptation to monochrome line art and screentones.
   - **Anime Illustrations** (`AnimeHeads`, `ah_`): evaluates performance on colored anime art.
   - **Overall Combined Split**.
4. **Visual Disagreement Panels:** Generates side-by-side composite grids in `results/benchmarks/visuals/` highlighting difficult cases (small background faces, screentones, profile angles).

### Running Benchmarks Locally
```bash
# Face benchmark (Catnip vs Fuyucchi, zymk9, and DeepGHS Face Nano)
pixi run benchmark-faces --sample 200 --device mps

# Head benchmark (Catnip vs DeepGHS Head Nano)
pixi run benchmark-heads --sample 200 --device mps

# Specify custom checkpoint and output dir
python scripts/eval/benchmark_faces.py \
    --target face \
    --catnip-model models/yolo26_stage1_body_head_face_m3.pt \
    --sample 500 \
    --device mps \
    --output-dir results/benchmarks_face
```

### Running on Google Colab (GPU)
`notebooks/catnipEval.ipynb` is the dedicated evaluation notebook: it verifies the R2 bucket, pulls the sliced test dataset and the trained model (`yolo26_stage1_body_head_face.pt`) from `catnip-canonical`, runs both the face and head benchmarks, and uploads `summary.md`, `metrics.json`, and the `visuals/*.jpg` comparison panels to `s3://catnip-canonical/results/benchmarks/{face,head}/`:
```bash
python scripts/eval/benchmark_faces.py \
    --target face \
    --catnip-model /content/catnip-data-local/models/yolo26_stage1_body_head_face.pt \
    --dataset-yaml /content/catnip-data-local/training/stage1_sliced/dataset.yaml \
    --device cuda \
    --sample 500 \
    --output-dir /content/catnip-data-local/results/benchmarks_face
```

For headless execution from `360 WSL`, `scripts/colab/launch_eval_360.sh` allocates the `catnip-eval` Colab GPU session and runs `scripts/colab/remote_eval_runner.py`, which synchronizes all artifacts to the same `results/benchmarks/` prefix.

---

## 6. Verification & Failure Modes

### Verifying Slicer Parity
```python
from src.config import settings
from src.training.slicing import slice_split

# Slice parameters must match training imgsz
assert settings.params.sahi.slice_height == settings.training.stage1.imgsz
assert settings.params.sahi.slice_width  == settings.training.stage1.imgsz
```

### Common Failure Modes & Fixes

| Symptom | Cause | Fix |
|---|---|---|
| MPS run crawls (~2.8 s/it @ batch 8) | Ultralytics clamps dataloader `workers=0` on MPS | `scripts/train/stage1.py` applies `_patch_mps_workers`. Check that logs report 4 workers. |
| MPS validation crashes on `torchvision::nms` | Missing MPS kernel in torchvision | Set `PYTORCH_ENABLE_MPS_FALLBACK=1` (set automatically in `stage1.py`). |
| MPS process killed with exit code 137 | macOS jetsam killed process due to Metal driver leak | Do not run multi-epoch single processes on MPS; use `pixi run train-stage1-m3-chain`. |
| S5cmd / Wrangler auth failure | Missing R2 credentials | Ensure `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, and `R2_ENDPOINT` are exported in `.env`. |
| Colab disconnected / timed out | Browser session closed | Use the headless runner on `360 WSL` (`scripts/colab/launch_360.sh`). |
