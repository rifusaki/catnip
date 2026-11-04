# Catnip: System Architecture & Design

> **Operations:** For step-by-step data prep, pre-slicing, and training commands, see [`docs/runbook.md`](runbook.md).

## 1. Objective

Create an automated computer vision pipeline to answer the question: *"Is Dungeon Meshi character Izutsumi present in this manga page, and where?"*

---

## 2. Machine Learning Architecture (Two-Stage Pipeline)

Due to extreme variations in character scale across comic layouts—ranging from full-page spreads to tiny background crowd faces—the system divides the task into a standard two-stage pipeline: **Localization (Detection)** followed by **Identification (Metric Learning)**.

```
Manga Page (Full Resolution)
         │
         ▼
┌───────────────────────────────────────────────┐
│ Stage 1: Sliced Localization (SAHI + YOLO26n) │
│ - Slices page into overlapping 640×640 patches│
│ - Detects all bodies, heads, and faces        │
│ - Merges predictions back to full-page coords │
└───────────────────────┬───────────────────────┘
                        │ Bounding Box Crops (Faces & Bodies)
                        ▼
┌───────────────────────────────────────────────┐
│ Stage 2: Character Re-ID (ResNet18 + GeM)     │
│ - 128-d L2-normalized embedding extraction    │
│ - FAISS IndexFlatIP cosine similarity         │
│ - Compares against Izutsumi seed embeddings   │
└───────────────────────┬───────────────────────┘
                        │ Matched Bounding Boxes & Confidence
                        ▼
           Tagged Izutsumi Instances
```

---

### 2.1 Label Strategy & Class Remapping

Source annotations in Label Studio use 6 fine-grained labels to retain full annotation fidelity. These are remapped at ingestion time according to stage needs:

| Label Studio Label | Description | Stage 1 Class (Detection) | Stage 2 Class (Re-ID) |
|---|---|---|---|
| `izutsumi_body` | Izutsumi's full body / torso | `body` (0) | `izutsumi` (0) |
| `izutsumi_head` | Izutsumi's head region | `head` (1) | `izutsumi` (0) |
| `izutsumi_face` | Izutsumi's tight face close-up | `face` (2) | `izutsumi` (0) |
| `other_body` | Another character's body | `body` (0) | `not_izutsumi` (1) |
| `other_head` | Another character's head | `head` (1) | `not_izutsumi` (1) |
| `other_face` | Another character's face | `face` (2) | `not_izutsumi` (1) |

- **Stage 1 Target:** 3 classes (`0: body`, `1: head`, `2: face`). Detects all candidate regions agnostic of character identity.
- **Stage 2 Target:** 2 classes (`0: izutsumi`, `1: not_izutsumi`). Distinguishes Izutsumi from all other characters.

---

### 2.2 Stage 1: Sliced Localization (SAHI + YOLO26)

- **Model:** `yolo26n` (Ultralytics YOLO26 Nano) pretrained on COCO, fine-tuned on the unified manga/anime detection dataset.
- **SAHI (Slicing Aided Hyper Inference):** Manga pages vary from 800×1200 to 2000×3000+ pixels. Slicing into overlapping 640×640 windows catches tiny faces without losing high-resolution detail or requiring panel segmentation.
- **Train/Inference Parity (Pre-Slicing):** The training pipeline pre-slices training images with the exact same SAHI parameters (`slice_height=640`, `slice_width=640`, `overlap_ratio=0.2`). Training at `imgsz=640` guarantees the model sees identical object scale and context distributions during both training and inference.
- **Training Setup:** Cosine LR schedule, MuSGD optimizer, patience=15, 150 epochs (or 40-epoch stratified subset runs).

---

### 2.3 Stage 2: Metric Learning & Re-Identification

- **Model:** ResNet18 backbone (ImageNet pretrained) + Generalized Mean (GeM) Pooling + Triplet Margin Loss.
- **Embedding Space:** Maps image crops to 128-dimensional L2-normalized embeddings.
- **Why Metric Learning over Standard Classification:** Allows open-set matching against few-shot reference "seeds" of Izutsumi without retraining when new reference art is added.
- **Matching:** FAISS `IndexFlatIP` computes cosine similarity between candidate crops and reference seed vectors. Candidates above the similarity threshold (default: 0.70) are classified as Izutsumi.

---

## 3. Infrastructure & Compute Strategy

| Tier | Environment | Role |
|---|---|---|
| **Dev & Local Prototyping** | M3 Mac Air (16 GB, Apple Silicon MPS) | Code development, pipeline testing, local stratified subset training (`train-stage1-m3-chain`). |
| **Heavy Remote Training** | **Colab L4 GPU (via 360 WSL)** or **Spot GPU VM (Vast.ai/RunPod RTX 4090)** | Stage 1 full training and Stage 2 deep metric learning. |
| **Storage Archive** | **Cloudflare R2 (`catnip-canonical`)** | Canonical cloud archive for datasets, weights, and rolling 10-minute training checkpoints ($0 egress). |
| **Edge Serving (Planned)** | Home Server (`360 WSL`, CPU / Intel XPU) | Containerized FastAPI inference server. |

---

## 4. Current Codebase Status

| Component | Path | Status | Notes |
|---|---|---|---|
| Configuration | `config/pipeline.yaml`, `src/config.py` | Active | Pydantic v2 + OmegaConf hierarchical settings with CLI dot-overrides. |
| Data Unification | `scripts/unify/*.py`, `src/preprocess/` | Active | Parsers for Label Studio, Manga109, AnimeHeadsv3 (COCO), and deepghs (YOLO). |
| Pre-slicing | `src/training/slicing.py` | Active | SAHI pre-slicing engine with bbox clipping and empty slice tracking. |
| Stage 1 Training | `scripts/train/stage1.py`, `m3_chain.sh` | Active | MuSGD, MPS worker patch, cache guards, one-epoch-per-process resume chain. |
| Stage 1 Model | `models/yolo26_stage1_body_head_face_m3.pt` | Active | Delivered Stage 1 checkpoint (mAP50 0.696 on 16k test split). |
| Stage 2 Training | `scripts/train/stage2.py` | Scaffold | Data preparation implemented; PyTorch Triplet training loop in development. |
| SAHI Extractor | `src/detection/sahi_extractor.py` | Active | Working SAHI inference wrapper with Non-Maximum Merging (NMM). |
| Inference CLI | `src/cli.py` | In Progress | Subcommands for `extract`, `embed`, and `match`. |
