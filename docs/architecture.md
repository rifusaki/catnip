# Catnip: System Architecture & Design

> **Operations:** For step-by-step data prep, pre-slicing, and training commands, see [`docs/runbook.md`](runbook.md).
> **Implementation Plan & Handoff:** For the Stage 1 v2 (`yolo26m` on L4, 97–98%+ precision) and Stage 2 Semi-Supervised Clustering implementation plan, see [`docs/implementation_plan.md`](implementation_plan.md).

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

- **Baseline Model (`v1`):** `yolo26n` (Ultralytics YOLO26 Nano, 2.5M params) pretrained on COCO, fine-tuned on the unified manga/anime detection dataset (`0.8823` face mAP@50, `0.8555` head mAP@50 on the 16k test split).
- **SAHI (Slicing Aided Hyper Inference):** Manga pages vary from 800×1200 to 2000×3000+ pixels. Slicing into overlapping 640×640 windows catches tiny faces without losing high-resolution detail or requiring panel segmentation.
- **Train/Inference Parity (Pre-Slicing):** The training pipeline pre-slices training images with the exact same SAHI parameters (`slice_height=640`, `slice_width=640`, `overlap_ratio=0.2`). Training at `imgsz=640` guarantees the model sees identical object scale and context distributions during both training and inference.
- **Stage 1 v2 Roadmap (Colab Pro L4 GPU — Target 97–98%+ Precision & Colored Art Generalization):**
  1. **Architecture Scale-Up:** Upgrade from `yolo26n` (2.5M) to **`yolo26m` (~20M params)** on Colab Pro L4 (24 GB VRAM, `batch=48–64`, `amp=True`, `cls_pw=1.0`).
  2. **Geometrically Verified Synthetic Pseudo-Labeling (Replacing Partial Labels):**
     - Eliminates negative BCE gradient contamination caused by `Manga109` lacking `head` labels and `AnimeHeads` lacking `face` and `body` labels (superior to sequential staged training, which suffers from catastrophic class forgetting).
     - **`head` on `Manga109`:** Generated via `deepghs/anime_head_detection` (`head_detect_v2.0_x_yv11`) + `Catnip-Stage1` WBF ensemble, accepted only when enclosing a GT `<face>` ($\text{IoS} \ge 0.80$) or sitting atop a GT `<body>`.
     - **`face` on `AnimeHeads`:** Generated via `deepghs/anime_face_detection` (`face_detect_v1.4_x`) + `zymk9_face` strictly inside GT `head` boxes ($\text{IoS} \ge 0.85$), transformed through a **4-parameter linear box-regression adapter** $(\Delta x_c, \Delta y_c, s_w, s_h)$ fitted on `Manga109`/`izutsumi` to align Danbooru face bounds to tight `Manga109`/`izutsumi` `<face>` conventions.
     - **`body` on `AnimeHeads`:** Generated via `deepghs/anime_person_detection` (`person_detect_v1.2_x`) verified to contain a GT `head` ($\text{IoS} \ge 0.70$); disembodied portrait crops without torsos are downsampled so training slices carry complete 3-class (`body`, `head`, `face`) supervision.
  3. **Color-to-Line-Art Bridging:** Enable `hsv_h=0.015`, `hsv_s=0.5`, `hsv_v=0.4` + `RandomGrayscale(p=0.25)` on colored illustrations so early backbone layers learn domain-invariant line/contour representations across B&W manga and colored fanart.
  4. **Dynamic Panel/Page Cropping & WBF:** Gate SAHI slicing in `src/detection/sahi_extractor.py` so pre-cropped manga panels ($\max(H, W) \le 800\text{px}$) run direct `640px` letterboxed inference while full pages run `640×640` SAHI slicing, followed by **Weighted Box Fusion (WBF)** and hierarchical containment filtering (`face` $\subset$ `head` $\subset$ `body`).

---

### 2.3 Stage 2: Semi-Supervised Multi-Prototype Character Clustering & Re-Identification

- **Evolution from Binary Triplet Loss to Semi-Supervised Clustering:**
  - *Limitations of Naive Binary Triplet Loss (`ResNet18` `izutsumi` vs `not_izutsumi`):* Treating `"not_izutsumi"` as a single class forces visually distinct characters (Laios, Marcille, Senshi, NPCs) together, while representing Izutsumi with a single unimodal centroid fails across her distinct visual modes *(standard cat-girl face, feral/chibi comedic reaction faces, hooded Ch. 31–33 panels, full-body tail silhouettes, and colored fanart vs. monochrome screentones)*.
- **Stage 2 Target Architecture (`src/reid/`):**
  1. **Domain-Pretrained Feature Extractor (`src/reid/model.py`):** Anime/manga-pretrained backbone (`deepghs/ccip`, `DINOv2-ViT-S/14`, or `ConvNeXt-Small` + Generalized Mean `GeM` Pooling) fine-tuned on L4 with **Supervised Contrastive Loss (`SupConLoss`) / Sub-Center ArcFace** into L2-normalized embeddings.
  2. **Multi-Granularity Instance Fusion:** Fuses embeddings from nested `[face, head, body]` detections belonging to the same character instance within a panel.
  3. **FAISS k-NN Graph + HDBSCAN Sub-Clustering (`src/reid/clustering.py`):**
     - Indexes all candidate character crops in `FAISS IndexFlatIP` and discovers natural density sub-clusters via **HDBSCAN** (or Leiden community detection).
     - Models Izutsumi as a **collection of verified sub-cluster medoids/prototypes** (covering normal, comedic/chibi, hooded, silhouette, and colored fanart modes) anchored by Must-Link (`izutsumi`) and Cannot-Link (`other`) seeds.
  4. **Contextual Panel & Page Co-Occurrence Constraints:**
     - *Same-Panel Cannot-Link:* Two distinct character bodies inside the same panel cannot both be assigned to Izutsumi.
     - *Same-Page Prior Propagation:* A high-confidence Izutsumi match in one panel boosts the matching prior for partial/back-view crops in adjacent panels on the same page.
  5. **Two-Tier Verification:** Auto-accepts core cluster members ($\text{sim} \ge \tau_{\text{core}}$) at **98%+ precision**, while surfacing ambiguous boundary sub-clusters for 1-click cluster-level confirmation.

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
