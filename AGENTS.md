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
| Model Benchmarks | `src/eval/`, `scripts/eval/benchmark_faces.py` | ✅ **Fully working.** Multi-model comparative benchmark for faces and heads (Catnip vs Fuyucchi, zymk9, DeepGHS). |
| Remote Eval Runner | `scripts/colab/remote_eval_runner.py`, `launch_eval_360.sh` | ✅ **Fully working.** Automated headless evaluation runner driven via 360 WSL / Colab CLI syncing artifacts to R2. v2 knobs: `CATNIP_EVAL_MODEL_KEY`, `CATNIP_EVAL_RESULTS_PREFIX`, `CATNIP_REPO_BRANCH` (defaults keep the v1 flow). |
| Presentation Notebook | `notebooks/catnipBenchmarkPresentation.ipynb` | ✅ **Fully working.** Light Mode presentation notebook with live 16k test metrics, domain breakdowns, and visual galleries. |
| Stage 2 ReID Modules | `src/reid/{model,dataset,loss,clustering}.py` | ✅ **Implemented (2026-11-06).** ConvNeXt/DINOv2+GeM 256-d embeddings, SupCon + Sub-Center ArcFace (K=5), HDBSCAN pseudo-cluster mining, multi-prototype PrototypeBank w/ Cannot-Link veto + panel/page constraints. Weight training runs on Colab; training loop verified end-to-end locally on MPS (0.07→0.91 separability on synth crops). |
| Stage 2 Training | `scripts/train/stage2.py` | ✅ **Rewritten (2026-11-06).** SupCon/ArcFace fine-tune over HDBSCAN-mined `not_izutsumi` pseudo-clusters; `--prepare-only` preserves legacy exit-0 prep. |
| CLI Interface | `src/cli.py` | ✅ **Wired (2026-11-06).** `extract` (gated SAHI v2), `embed` (linked instance fusion, `embed_meta.json` provenance), `match` (two-tier clustering + review HTML), `pipeline`. Global + subcommand `--override` merge correctly. |
| Cross-Dataset Pseudo-Labeling | `scripts/unify/pseudo_label.py`, `src/preprocess/box_calibration.py` | ✅ **Fully working (2026-11-06).** Rule A/B head verification, calibrated faces, verified bodies, deterministic 10% headshot draw; enriched trees never mutate inputs, rerun-converging. Adapter fitted on real data: `config/face_box_adapter.json` (10,884 pairs, recall@IoU0.75 0.455→0.523). |
| v2 Data Chain Orchestrator | `runs/v2-pipeline/pipeline_v2.sh` (SSD) | ✅ **Ran to completion (2026-11-06).** Idempotent/resumable chain: enrichment → izutsumi passthrough → unify+slice → tar → R2 upload (per-part markers). Survived one mid-run SSD ejection untouched. |

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

### Gotcha 6: Dataset Annotation Boundaries in Multi-Domain Benchmarks (Head vs Face)
* **Symptom:** Evaluating `--target face` on anime illustration datasets (`ahv1`, `ahv2`, `ahv3`, `ahaf`) yields $0.000$ mAP, and external anime face detectors hallucinate dozens of false positives on non-annotated background elements.
* **Root Cause:** In Catnip Stage 1 data prep, AnimeHeads datasets are **head-only datasets** (`class 1: head`) and possess **0 ground truth face annotations (`class 2`)**.
* **Fix:** Always benchmark **face** detection on the annotated Manga Line-Art domain (Manga109 & Izutsumi, 24,785 GT faces). For **head** detection, ground truth exists across both manga and anime illustration domains.

### Gotcha 7: Domain Prefix Matching
* **Symptom:** Slices from `ah_yolo` and `ah_coco` get dumped into `"general"` instead of `"anime_illustration"`.
* **Fix:** Slices use filenames prefixed with `ahv1_`, `ahv2_`, `ahv3_`, and `ahaf_`, not simply `ah_`. Always match all prefix variants (`("ahv", "ahaf", "ah_", "ah-", "anime", "heads")`) when stratifying domains.

### Gotcha 8: Visual Comparison Selection Bias
* **Symptom:** The visual comparison gallery fills entirely with images where `Ground Truth (0)` and `Catnip (0)`, while external peer models hallucinate 35–142 boxes.
* **Root Cause:** Sorting visual candidates purely by model prediction count variance (`np.std(pred_counts)`) strongly prioritizes background slices with 0 GT boxes where peers hallucinate heavily.
* **Fix:** In `scripts/eval/benchmark_faces.py`, candidate pools are strictly filtered to slices with `len(gt_boxes) > 0` (`gt_positive_candidates`) before ranking.

### Gotcha 9: Jupyter Kernel Isolation on macOS
* **Symptom:** Headless notebook execution via `nbclient` defaults to system/Homebrew Python (`/opt/homebrew/bin/python3.14`) due to user-level kernelspecs in `~/Library/Jupyter/kernels/python3`.
* **Fix:** `scripts/eval/execute_presentation_notebook.py` uses `CurrentEnvKernelSpecManager` to bind execution directly to `sys.executable`, guaranteeing portability across macOS, WSL, and Linux VMs.

### Gotcha 10: deepghs Teacher Checkpoint Rot (no x-sized face/person variants)
* **Symptom:** The roadmap's `face_detect_v1.4_x` and `person_detect_v1.2_x` URLs return 404 — deepghs only publishes `n`/`s` sizes in those two repos (verified via the HF repo trees 2026-11-06).
* **Fix:** The catalog (`src/eval/adapters.py`) pins `deepghs_face_s` (`face_detect_v1.4_s`) and `deepghs_person_s` (`person_detect_v1.2_s`); `deepghs_head_x` (`head_detect_v2.0_x_yv11`) does exist and stays. Rule of thumb: before blaming the pipeline on a teacher download failure, `curl -I` the HF tree — these repos change layouts. Side benefit: s-sized teachers are MPS-feasible locally.

### Gotcha 11: stage2.py `--override` must ACCUMULATE (append), not last-one-wins
* **Symptom:** `scripts/train/stage2.py` originally parsed `--override` with `nargs="*"` and no `append`, so the repo-wide convention of repeating `--override KEY=VALUE` per flag silently kept only the LAST flag (e.g. an `epochs=1` smoke actually trained /100).
* **Fix:** `--override` is now `action="append"`; `test_stage2_training.py::test_stage2_cli_accumulates_repeated_override_flags` pins it. Same class of bug, cli-wide: global `--override`/`-v` live on distinct dests and merge in `main()` (subcommand entries applied last, so they win collisions).

### Gotcha 12: Teacher Detector Construction is Per-Image-Expensive — Cache It
* **Symptom:** `run_teachers` constructing its detector per image made the 2,075-page adapter calibration pass spend minutes per page just reloading weights (and the whole torch.hub repo for `zymk9_face`).
* **Fix:** `pseudo_label.py` caches detectors per `(model_key, target, device, path)` for the process lifetime; the full calibration dropped to ~13 min on MPS. New teacher consumers should reuse the same seam.

### Gotcha 13: wrangler's 300 MiB Object Cap vs. the 13 GB v2 Archive
* **Symptom:** `npx wrangler r2 object put` hard-fails above **300 MiB** ("Wrangler only supports uploading files up to 300 MiB") — a 13 GB `stage1_v2_sliced.tar` cannot go up in one object, and even naive multi-GB splits die.
* **Fix:** The pipeline initially split archives into ~290 MB `.part_*` chunks due to wrangler CLI limits. `notebooks/catnipConsolidateR2Tar.ipynb` (and `scripts/colab/consolidate_r2_tar.py`) uses S3 Multipart Upload via boto3 to join all 46 parts directly on a CPU Colab instance and safely replace them with the unified `compressed/stage1_v2_sliced.tar` in R2. `remote_runner.py` natively supports both the unified archive (direct download) and split parts fallback.

### Gotcha 14: SSD Ejections + exec-argv Limits in the v2 Pipeline
* **Symptom:** A mid-run SSD ejection fails the chain loudly at the next artifact check (enrichment writers only touch disk at each source's END, so no partial trees result). Separately, `cat`-ing ~74k label files via a glob blows the exec argv limit (`Argument list too long`, exit 126).
* **Fix:** `runs/v2-pipeline/pipeline_v2.sh` is idempotent and resumable — every step skips on its manifest/marker file (per-part upload markers included), and the class tally uses `find | xargs -0 cat`. Re-run the script after any ejection; it converges to the same state (`state.log` carries DONE/SKIP/FAIL markers). Running the chain under `caffeinate -i` avoids sleep-induced bus unmounts.

### Gotcha 15: Stage 1 v2 Sliced Dataset Directory Resolution on Colab
* **Symptom:** `scripts/colab/remote_runner.py` extracting `compressed/stage1_v2_sliced.tar` puts the dataset in `/content/catnip-data-local/training/stage1_v2_sliced`, but `stage1.py` historically looked only for `training/stage1_sliced` or `training/stage1`, exiting with code 1 (`No dataset.yaml found at /content/catnip-data-local/training/stage1_sliced or /content/catnip-data-local/training/stage1`).
* **Fix:** `stage1.py` now supports `--dataset <path>` directly and auto-discovers `stage1_v2_sliced` if `stage1_sliced` is absent. `remote_runner.py` forwards `--dataset` and `--override paths.stage1_sliced_dir={dataset_dir}` to both phase 1 and golden polish.

### Gotcha 16: MuSGD cuBLAS Strided Batched GEMM Incompatibility & L4 Batch Sizing
* **Symptom:** During step 5 of epoch 1 on CUDA, training crashes with `RuntimeError: CUDA error: CUBLAS_STATUS_NOT_SUPPORTED when calling cublasGemmStridedBatchedEx(..., CUDA_R_16BF, ...)`. Separately, `batch=48` triggers `WARNING ⚠️ CUDA out of memory with batch=48. Reducing to batch=24 and retrying`.
* **Root Cause:** In Ultralytics `muon.py`, `zeropower_via_newtonschulz5` casts tensors to `bfloat16` and executes `torch.baddbmm` on non-contiguous transposed tensors, which modern cuBLAS on Ada Lovelace (L4) rejects with status not supported. Furthermore, `yolo26m` (~20M params at `imgsz=640` with 306 max detections) requires ~21.8 GB VRAM at `batch=24`, exceeding 24 GB capacity if initialized at `batch=48`.
* **Fix:** Use `optimizer: AdamW` in `config/pipeline.yaml` (uses PyTorch's native `fused=True` CUDA kernels, matching `lr0=0.001` and `weight_decay=0.002`). `stage1.py` additionally patches `zeropower_via_newtonschulz5` with float32 contiguous layout as defense-in-depth, and `batch` is pinned to `24` on L4.

---

## 4. Storage & Data Conventions

- **Canonical Archive:** Cloudflare R2 bucket `catnip-canonical` (`s3://catnip-canonical`) with $0 egress. Local R2 access via `npx wrangler r2 object put/get` (300 MiB per-object cap — see Gotcha 13) or `rclone r2:` (multipart, authenticated 2026-11-06).
- **Local Data Dir:** `catnip-data/` (or `/Volumes/rifuSSD/catnip-data`).
- **Tarball for Cloud Instances:** `compressed/stage1_sliced.tar` (~14 GB) contains the pre-sliced v1 training dataset for instant hydration on remote VMs.
- **v2 Dataset (2026-11-06):** `compressed/stage1_v2_sliced.tar` (13 GB, 46 × ~290 MB `.part_*` objects in R2 — the runner reconstructs them) holds the fully 3-class enriched + re-sliced dataset; local copies on the SSD at `training/stage1_v2_sliced/` and `compressed/stage1_v2_sliced.tar(.part_*)`.
- **v2 Enriched Staging:** `data/staging_enriched/{izutsumi,manga109,ah_yolo,ah_coco}` on the SSD — per-source enriched trees (symlinked images, original + synthetic labels, `manifest.json` per source). Pristine `data/staging/` was never mutated and stays the source of truth for reruns.
- **Face Box Adapter:** `config/face_box_adapter.json` (committed) — fitted from 10,884 aligned pairs (izutsumi + strided 2,000-page manga109 sample), held-out recall@IoU0.75 0.455 → 0.523.
- **Base Weights:** `models/yolo26n.pt` (v1) and `models/yolo26m.pt` (v2, uploaded to R2 2026-11-06, md5-verified) exist on the SSD and in R2 for fresh Colab starts.
- **Rolling Checkpoints:** Remote runners push `runs/detect/<run>/weights/last.pt` to `s3://catnip-canonical/runs/<run>-last.pt` every 10 minutes. Training resumes seamlessly if interrupted.

---

## 4.1 Stage 1 v2 Enriched Dataset — Deliverable (2026-11-06)

Produced by `runs/v2-pipeline/pipeline_v2.sh` (local MPS, ~5.5 h total):

| Source | Seen | Kept | Headshot-only (10% draw kept) | Added body | Added head | Added face |
|---|---|---|---|---|---|---|
| manga109 | 10,071 | 10,071 | — | — | **90,872** | — |
| ah_yolo | 65,191 | 55,731 | 10,478 (kept 1,018) | **136,273** | — | **161,348** |
| ah_coco | 8,157 | 7,816 | 383 (kept 42) | **18,961** | — | **18,052** |
| izutsumi | 75 (passthrough) | 75 | — | — | — | — |

- Unified + SAHI re-sliced (80/10/10, seed 42, train-8-workers): **train 123,925 / val 15,945 / test 15,021 slices** from 73,693 kept originals.
- Enriched label class totals (all sources): **body 312,608 · head 332,343 · face 298,164** — all three classes fully populated across monochrome manga AND colored anime illustration; the partial-label BCE contradiction (roadmap §7.1) is resolved in the training data itself.

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

# Comparative Model Benchmarks
pixi run benchmark-faces --sample 200        # Benchmarks face detection (Catnip, Fuyucchi, zymk9, DeepGHS)
pixi run benchmark-heads --sample 200        # Benchmarks head detection (Catnip vs DeepGHS Head Nano)
pixi run benchmark-faces --download-only     # Downloads & caches all external benchmark weights
pixi run benchmark-presentation              # Headlessly executes & embeds results into catnipBenchmarkPresentation.ipynb
pixi run visualize-gt                        # Generates visual Ground Truth enrichment gallery into docs/gt_gallery & results/gt_gallery

# Pseudo-labeling & adapter calibration (v2 data chain; teachers cached per process — Gotcha 12)
pixi run python scripts/unify/pseudo_label.py --target-section calibrate --device mps --adapter config/face_box_adapter.json
pixi run python scripts/unify/pseudo_label.py --target-section manga109 --device mps --catnip-model <v1 .pt> --limit 30 --dry-run
pixi run python scripts/unify/pseudo_label.py --target-section anime --device mps --adapter config/face_box_adapter.json --limit 30 --dry-run
# Full v2 chain (enrichment -> izutsumi passthrough -> unify+slice -> tar -> R2 parts), idempotent/resumable:
bash /Volumes/rifuSSD/catnip-data/runs/v2-pipeline/pipeline_v2.sh        # caffeinate -i recommended

# Colab runners (360 WSL) — v2 profile is env-driven; defaults are the v2 values:
bash scripts/colab/launch_360.sh             # 4.2: yolo26m@L4: 75 epochs + 10 polish (CATNIP_STAGE1_* knobs)
bash scripts/colab/launch_eval_360.sh        # 2.1: fixed harness re-benchmarks v1 model into canonical results
CATNIP_EVAL_MODEL_KEY=models/yolo26m_stage1_v2.pt CATNIP_EVAL_RESULTS_PREFIX=results/benchmarks_v2 bash scripts/colab/launch_eval_360.sh  # 4.3 variant
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

### Comparative Benchmark Deliverables (CUDA Full Test Split, 15,993 Images)

#### Face Detection Benchmark (Manga Line-Art Domain, 24,785 GT Faces)
| Model | Architecture | Params | Weights | mAP@50 | mAP@50-95 | Recall | Latency |
|---|---|---|---|---|---|---|---|
| **Catnip-Stage1** | YOLO26n (640px) | **2.5M** | **5.4 MB** | **0.8823** | **0.5676** | **0.9040** | **14.1 ms** |
| **deepghs_face_nano** | YOLOv8n (640px) | 3.0M | 6.2 MB | 0.6429 | 0.2367 | 0.7053 | 9.9 ms |
| **zymk9_face** | YOLOv5s (640px) | 7.2M | 14.8 MB | 0.5326 | 0.2408 | 0.5928 | 12.2 ms |
| **fuyucchi_face** | YOLOv8x6 (1280px) | 97.3M | 195.4 MB | 0.4579 | 0.1300 | 0.5743 | 76.8 ms |

#### Head Detection Benchmark (15,993 Images, 27,414 GT Heads)
| Model | Architecture | Params | Weights | Overall mAP@50 | Manga mAP@50 | Anime mAP@50 | Latency |
|---|---|---|---|---|---|---|---|
| **Catnip-Stage1** | YOLO26n (640px) | **2.5M** | **5.4 MB** | **0.8555** | **0.9101** | **0.8000** | **13.5 ms** |
| **deepghs_head_nano** | YOLO11n (640px) | 2.6M | 5.5 MB | 0.4140 | 0.0096 | 0.8000 | 11.7 ms |

---

## 7. Next-Gen Architecture Roadmap (Stage 1 v2 & Stage 2 Clustering)

> **Detailed Implementation Plan & Engineering Handoff:** See [`docs/implementation_plan.md`](docs/implementation_plan.md) for phased tasks, teacher model URLs, box-regression math, L4 hyperparameters, and module contracts.

### 7.1 Stage 1 v2: Reaching 97–98%+ Precision & Cross-Domain Generalization (Colab L4 GPU)
- **Compute & Model Upgrade (`yolo26m`):** Moving primary training to **Colab Pro L4 (24 GB VRAM)** enables scaling from `yolo26n` (2.5M params) to **`yolo26m` (~20M params)** at `imgsz=640`, `batch=48–64`, `amp=True`, `cls_pw=1.0`.
- **Cross-Dataset Synthetic Pseudo-Labeling (Solving Partial-Label Contradiction):**
  - *Problem:* `Manga109` only annotates `body` (0) + `face` (2) (missing `head`), while `AnimeHeads` (`ah_yolo`, `ah_coco`) only annotates `head` (1) (missing `body` + `face`). Unannotated classes receive negative BCE gradients during joint training, teaching a domain shortcut (suppressing `head` on manga and suppressing `face`/`body` on colored art).
  - *Why Pseudo-Labeling Beats Staged Training:* Sequential staged training causes catastrophic forgetting of unannotated classes at each stage (`izutsumi` is only 75 pages, too small to recover general 3-class weights).
  - *Teachers & Geometric Verification:*
    - **`head` (`class 1`) on `Manga109`:** `deepghs/anime_head_detection` (`head_detect_v2.0_x_yv11`) + `Catnip-Stage1` WBF ensemble. Accepted iff the predicted head encloses a GT `Manga109` `<face>` ($\text{IoS} \ge 0.80$) or overlaps the top 35% of a GT `<body>` box. (`deepghs_head` shares the exact same head box convention as `ah_yolo`).
    - **`face` (`class 2`) on `AnimeHeads`:** `deepghs/anime_face_detection` (`face_detect_v1.4_s` — *correction 2026-11-06: deepghs publishes no x-sized face/person checkpoints; the roadmap's `_x` URLs 404; Gotcha 10*) + `zymk9_face` accepted iff strictly inside a GT `head` box ($\text{IoS} \ge 0.85$), passed through a **4-parameter linear box-regression adapter** $(\Delta x_c, \Delta y_c, s_w, s_h)$ fitted on `Manga109`/`izutsumi` `(pred_face, gt_head) -> gt_face` pairs to align loose Danbooru face boxes to tight `Manga109`/`izutsumi` `<face>` bounds. *(Status: fitted 2026-11-06 — `config/face_box_adapter.json`, 10,884 pairs, held-out recall@IoU0.75 0.455 → 0.523.)*
    - **`body` (`class 0`) on `AnimeHeads`:** Keep `body` (critical for Izutsumi back/silhouette panels and Stage 2 same-panel constraints). Generate via `deepghs/anime_person_detection` (`person_detect_v1.2_s` — *see Gotcha 10 correction*) verified to contain at least one GT `head` ($\text{IoS} \ge 0.70$). Filter/downsample headshot-only `AnimeHeads` crops where no torso exists ($\text{IoU}(\text{body}, \text{head}) > 0.85$) so training slices have complete 3-class (`body`, `head`, `face`) annotations. *(Status: full chain delivered 2026-11-06 — §4.1.)*
- **Color-to-Line-Art Bridging:** Re-enable HSV augmentation (`hsv_h=0.015`, `hsv_s=0.5`, `hsv_v=0.4`) and apply `RandomGrayscale(p=0.25)` on colored `AnimeHeads` slices so the backbone learns shared structural contour features across monochrome manga and colored fanart.
- **Dynamic Panel vs. Page Inference & WBF:**
  - In `src/detection/sahi_extractor.py`, gate slicing by input resolution: full pages ($\max(H, W) > 800\text{px}$) use normalized `640×640` SAHI slicing; pre-cropped individual manga panels ($\max(H, W) \le 800\text{px}$) use direct `640px` letterboxed inference.
  - Apply **Weighted Box Fusion (WBF)** and hierarchical anatomy filtering (requiring `face` detections below `conf=0.70` to lie inside a detected `head` or `body` box with $\text{IoS} \ge 0.75$) plus calibrated per-class thresholds (`conf ≈ 0.52–0.58`) to achieve **97–98%+ precision**.

### 7.2 Stage 2: Semi-Supervised Multi-Prototype Clustering (Replacing Binary Triplet Loss)
- **Why Semi-Supervised Clustering Replaces Binary `ResNet18` Triplet Loss:**
  1. `"not_izutsumi"` is a heterogeneous set of dozens of recurring characters (Laios, Marcille, Senshi, etc.), not a single visual class.
  2. Izutsumi is strongly **multi-modal** across manga and fanart *(stoic cat-girl face, wide-eyed feral/gremlin chibi reactions, hooded Ch. 31–33 appearances, full-body tail/silhouette crops, and colored fanart vs. B&W screentones)*—a single centroid with a fixed `0.7` cosine threshold either misses stylized modes or admits false positives.
- **Target Stage 2 Design (`src/reid/`):**
  1. **Domain-Pretrained Backbone (`src/reid/model.py`):** Use an anime/manga-pretrained feature extractor (`deepghs/ccip`, `DINOv2-ViT-S/14`, or `ConvNeXt-Small` + `GeM` pooling) fine-tuned on L4 with **Supervised Contrastive Loss (`SupConLoss`) / Sub-Center ArcFace** producing L2-normalized embeddings.
  2. **Multi-Granularity Instance Fusion:** Link nested Stage 1 detections (`face` $\subset$ `head` $\subset$ `body`) from the same character instance into a confidence-weighted fused descriptor.
  3. **FAISS + HDBSCAN / Graph Label Propagation (`src/reid/clustering.py`):**
     - Build a `FAISS IndexFlatIP` k-NN graph over all extracted crops and partition into density sub-clusters via **HDBSCAN** (or Leiden community detection).
     - Represent Izutsumi via **multiple sub-cluster medoids/prototypes** (capturing her normal, chibi/feral, hooded, body, and colored fanart modes) seeded by Must-Link (`izutsumi`) and Cannot-Link (`other`) anchors from Label Studio.
     - Enforce **Contextual Panel/Page Constraints**: *Same-Panel Cannot-Link* (two distinct character bodies in one panel cannot both be Izutsumi) and *Same-Page Prior Boost* (high-confidence match in Panel 1 boosts matching prior for side/back views in adjacent panels on the same page).
     - Two-tier decision boundary: auto-tag core members ($\text{sim} \ge \tau_{\text{core}}$) at **98%+ precision**, and surface boundary sub-clusters for 1-click cluster-level human verification.

### 7.3 Open-Source Packaging & Naming (Status: Pending Decision)
Both Stage 1 (cross-domain B&W manga + colored artwork `body`/`head`/`face` YOLO26m detector) and Stage 2 (semi-supervised manga/artwork character clustering) generalize beyond *Dungeon Meshi* for public release (Hugging Face Hub weights + Gradio Space + CLI exporters for `.cbz`, Eagle, Hydrus, Kohya, and Label Studio).
- **Naming Candidates (Pending Final Decision):**
  - *Option A (Feline / Lore):* Umbrella **`Catnip`** (`catnip-vision`) $\rightarrow$ Stage 1 **`Catseye`** / **`Tapetum`** (`tapetum-yolo26m`) + Stage 2 **`Clowder`** (`clowder-reid`), checkpoint codename `asebi`.
  - *Option B (Manga Craft):* **`Koma`** (`koma-detect` + `koma-cluster`).
  - *Option C (Fandom-First):* **`blorbo`** (`pip install blorbo`).


