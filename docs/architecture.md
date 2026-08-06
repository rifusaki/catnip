# Catnip: Project Architecture & Roadmap Proposal

> **Status:** For implementation details, see [`docs/implementation-plan.md`](implementation-plan.md).

## 1. Objective

Create an ML pipeline to identify Dungeon Meshi character Izutsumi.

## 2. Machine Learning Architecture (Two-Stage Pipeline)

The system uses an industry-standard two-stage pipeline: detect first, then identify.

### Label Strategy

Label Studio maintains 4 fine-grained labels for annotation fidelity:

| Label             | Meaning                                    | Stage 1     | Stage 2          |
|-------------------|--------------------------------------------|-------------|------------------|
| `izutsumi_body`   | Izutsumi's full body or torso              | body (0)    | izutsumi (0)     |
| `izutsumi_head`   | Izutsumi's head region (renamed from old `izutsumi_face`) | body (0)    | izutsumi (0)     |
| `izutsumi_face`   | Izutsumi's face close-up (NEW, tighter)    | face (1)    | izutsumi (0)     |
| `other_body`      | Another character's full body              | body (0)    | not_izutsumi (1) |
| `other_head`      | Another character's head (renamed from old `other_face`) | body (0)    | not_izutsumi (1) |
| `other_face`      | Another character's face (NEW, tighter)    | face (1)    | not_izutsumi (1) |

**The 6 labels** stay in Label Studio permanently. Remapping to 2 classes (`body`/`face` for Stage 1, `izutsumi`/`not_izutsumi` for Stage 2) happens only at model ingestion time via `config/pipeline.yaml` → `labels:` mappings.

### Stage 1: Localization via SAHI + YOLO26

- **Purpose:** Find "where" bodies and faces are, regardless of *who* the character is.
- **Model:** YOLO26n finetuned on 3 classes (`body`, `head`, `face`). YOLO26 is the official Ultralytics successor to YOLO11 (Jan 2026), offering NMS-free detection, MuSGD optimizer, and better accuracy-per-param.
- **SAHI:** Slices full manga pages into overlapping 640×640 patches at inference. Catches both tiny faces in crowd scenes and massive full-page characters without panel extraction.
- **Pre-sliced training (train/inference parity):** The training set is pre-sliced with the **same** SAHI parameters via `scripts/unify/stage1.py --slice` (slicer module: `src/training/slicing.py`). YOLO is trained at `imgsz=640` so each training sample is byte-identical to what the model sees during SAHI sliced prediction. See `docs/stage1-runbook.md` for the full pipeline and GCS layout.
- **Training:** Finetuned from `yolo26n.pt` at imgsz=640, patience=0 (per Fuyucch1's anime face findings), ~300 epochs on Colab T4.
- **Target:** mAP50 ≥ 0.85 on body+face detection.
- **Output:** Cropped bounding boxes of all faces/bodies + per-image JSON metadata.
- **Config:** See `config/pipeline.yaml` → `sahi:` block and `labels.stage1`.

### Stage 2: Re-Identification & Metric Learning (PyTorch)

- **Purpose:** Identify "who" the character is from Stage 1 crops.
- **Model:** ResNet18 backbone (ImageNet pretrained) + GeM Pooling + Triplet Loss. Lightweight enough for M3 prototyping, powerful enough for the task. ViT is overkill for a hobby-scale dataset.
- **Why Triplet Loss over ArcFace:** Better for small identity counts (2 classes), simpler implementation, embeddings are directly compatible with cosine similarity (no margin angle conversion needed).
- **Training:** PyTorch `RandomIdentitySampler` (P=8, K=4), hard triplet mining within batch, 100 epochs on Colab T4.
- **Training data mix:** Ground truth crops from Label Studio bounding boxes + detector crops from Stage 1 output (ensures the ReID model sees cropper noise at training time).
- **Matching:** FAISS IndexFlatIP on L2-normalized embeddings → cosine similarity → threshold (default 0.7) → optional LogisticRegression re-ranking.
- **Target:** Rank-1 accuracy ≥ 0.80, mAP ≥ 0.75.
- **Config:** See `config/pipeline.yaml` → `metric_learning:` block and `labels.stage2`.

## 3. Software Engineering & Backend

Move the logic out of notebooks to demonstrate production-ready engineering.

- **CLI Application:** `main.py` with `argparse` subcommands (`extract`, `embed`, `match`, `pipeline`). All params overridable via `--override params.sahi.confidence_threshold=0.5` dot-notation.
- **REST API (Backend):** FastAPI server in `src/api/`. Endpoint `POST /api/v1/match` accepts a manga page image and returns JSON bounding boxes of detected Izutsumi instances.
- **Batch Processing:** PyTorch `Dataset` and `DataLoader` for efficient image processing.
- **Experiment Tracking:** CSV logging in `runs/` directory (epoch, loss, mAP, Rank-1).

## 4. Infrastructure & Compute Strategy

Three-tier compute distribution reflecting professional ML engineering patterns:

| Tier | Environment | Role |
|------|------------|------|
| **Dev & Prototyping** | M3 Mac Air (MPS) | Code development, pipeline testing, local sanity-check training |
| **Heavy Training** | **Kaggle Notebook (T4×2 GPU, 30 h/week)** — primary; Google Colab (T4 GPU) — fallback | Stage 1 YOLO26 training (300 epochs in 12 h chunks), Stage 2 triplet loss training (100 epochs) |
| **Edge Serving** | Home Server (CPU, ONNX) | Production inference via Dockerized FastAPI with ONNX-exported models |

- **Storage:** Google Cloud Storage bucket `catnip-data` is the *canonical archive*; Kaggle Datasets (`catnip-stage1-sliced`, `catnip-stage1-output`) are the *training cache*.  The pixi task `kaggle-sync` (run on the dev machine after `scripts/unify/stage1.py --slice`) keeps both in sync.  Cloudflare R2 migration deferred until home server API goes live (avoids egress costs for Label Studio + API traffic).
- **Label Studio:** Self-hosted at `label.rifusaki.com` (project 2). Images imported from GCS URLs. Exports JSON with bounding box annotations.
- **Why Kaggle over Colab:** 30 h/week hard cap with a visible counter replaces Colab's arbitrary, unannounced throttling.  Sequential dataset versioning (`kaggle datasets version`) makes resumed training sessions simpler than the Colab tarball re-upload dance.  12 h session cap is the trade-off, handled by `scripts/train/stage1.py --resume` plus publishing the run to `catnip-stage1-output` at session end.

## 5. Codebase Status

### Current State (June 2026)

| Directory/File | Status | Notes |
|---------------|--------|-------|
| `config/pipeline.yaml` | ✅ Active | Needs extension: `sahi:`, `metric_learning:`, `labels:` blocks |
| `src/config.py` | ✅ Active | Pydantic models; needs `SahiParams`, `Labels`, `MetricLearningParams` extensions |
| `src/convert_labels.py` | ✅ Active | Label Studio → YOLO; needs 4→2 class remap function and config-driven CLASS_MAP |
| `src/detection/sahi_extractor.py` | 🚧 Skeleton | Working SAHI wrapper; needs config integration, face/body separation, JSON metadata |
| `src/training/preparation.py` | ✅ Active | YOLO dataset prep; needs `prepare_triplet_dataset()` for Stage 2 |
| `src/output/output.py` | ⚠️ Has bugs | `save_inference_results` reusable; broken import at line 7 |
| `src/reid/model.py` | ❌ Empty | Must implement ResNet18 + GeM + TripletLoss |
| `src/reid/dataset.py` | ❌ Empty | Must implement TripletDataset + RandomIdentitySampler |
| `src/reid/matcher.py` | ❌ Empty | Port cosine similarity + FAISS logic (retrieve old `query.py` logic from git history) |
| `src/api/` | ❌ Empty | FastAPI server (post-MVP, not in current plan) |
| `notebooks/catnip.ipynb` | 📦 Archive | Obsolete once CLI exists; keep for reference |
| `notebooks/catnipColab.ipynb` | 🔄 Evolve | Becomes the Colab training runbook |
| `notebooks/catnipLocal.ipynb` | 🗑️ Deleted | Already removed from disk; content merged into catnip.ipynb |

### Deleted / No Longer Exist
| Path | Status |
|------|--------|
| `src/recognition/` | 🗑️ Deleted (old TF/MobileNetV2 query pipeline) |
| `src/preprocess/` | 🗑️ Deleted (panel extraction replaced by SAHI) |
| `modules/coreMPE/` | 🗑️ Deleted (panel extraction dependency) |

### Known Bugs (Phase 1)
- `src/training/preparation.py:173` — dead `print()` outside function body
- `src/output/output.py:7` — broken import from deleted module
- `src/training/preparation.py:36-60` — commented-out `prepare_data()` cruft

### GCS Bucket Legacy
See `docs/implementation-plan.md` Phase 1, task 1.6. The bucket `catnip-data` contains stale training runs (`runs/izutsumi*`, `runs/0.11.*`), old results (`results/`), and orphaned annotation versions (`data/annotations/v01-v12`). Phase 1 includes reorganization.

## 6. Phased Implementation Plan

Detailed tasks, success metrics, and edge cases are in [`docs/implementation-plan.md`](implementation-plan.md). High-level phases:

| Phase | Scope | Deliverables |
|-------|-------|-------------|
| **1. Config & Data Pipeline** | YAML/Pydantic sync, label conversion, triplet dataset prep, bug fixes, edge cases, GCS bucket cleanup | Updated config, `convert_labels_stage1()`, `prepare_triplet_dataset()`, clean GCS layout |
| **2. Stage 1: SAHI Detection** | SAHI extractor finalization, YOLO26 training on 2 classes, ground truth crop extraction | `yolo26_stage1_face_body.pt`, `sahi_extractor.py`, `extract_gt_crops.py` |
| **3. Stage 2: ReID Model** | ResNet18 + GeM + TripletLoss model, dataset, matcher, training loop | `reid_stage2_best.pth`, `model.py`, `dataset.py`, `matcher.py` |
| **4. Integration & CLI** | `main.py` CLI, training scripts, Colab notebook update | `main.py`, `scripts/train/stage1.py`, `scripts/train/stage2.py`, updated Colab notebook |

### Success Metrics
| Metric | Target |
|--------|--------|
| Stage 1 mAP50 (face+body) | ≥ 0.85 |
| Stage 2 Rank-1 accuracy | ≥ 0.80 |
| Stage 2 mAP | ≥ 0.75 |
| Pipeline latency | &lt; 30s per page |

## 7. Deployment Roadmap (Post-MVP)

After the core pipeline is working:
1. **ONNX Export:** Convert both YOLO26 and ResNet18 models to ONNX for CPU-efficient inference on the home server.
2. **FastAPI Backend:** Implement `src/api/` with `/api/v1/match` endpoint.
3. **Dockerize:** Multi-stage Dockerfile → deployed on home server.
4. **Cloudflare R2 Migration:** Move storage from GCS to R2 when home server egress becomes the dominant cost.
5. **CI/CD:** GitHub Actions for linting + tests on PR.
