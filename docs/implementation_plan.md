---
status: in-progress
phase: 4
updated: 2026-11-08
---

# Catnip Next-Gen Implementation Plan & Engineering Handoff

> **Companion Docs:** [`AGENTS.md`](../AGENTS.md) · [`docs/architecture.md`](architecture.md) · [`docs/runbook.md`](runbook.md)

## Goal
Upgrade Catnip Stage 1 to **97–98%+ precision** across both monochrome manga panels and colored artwork on **Colab Pro L4 (`yolo26m`)** via geometrically verified cross-dataset pseudo-labeling, color-to-line-art bridging, dynamic panel/page inference, and WBF, followed by implementing **Stage 2 Semi-Supervised Multi-Prototype Character Clustering** in `src/reid/`.

---

## Context & Decisions

| Decision | Rationale | Source |
|---|---|---|
| **Re-generate benchmark visuals with post-`9f0ff6b` harness** | Existing `results/benchmarks/face/visuals/*.jpg` were generated at `15:50` (before commit `9f0ff6b` at `17:44`), when the ranker sorted purely by prediction variance without filtering `len(gt_boxes) > 0`, selecting 12 `AnimeHeads` posters where `GT=0` and `Catnip=0`. | `ref:b6f94a41-07e0-4ef3-9467-e193815270e8` |
| **Geometrically verified synthetic pseudo-labeling over sequential staged training** | `Manga109` lacks `head` (`class 1`) and `AnimeHeads` lacks `body` (`class 0`) + `face` (`class 2`). Sequential staged training causes catastrophic forgetting of unannotated classes via negative BCE gradients, and `izutsumi` (75 pages) is too small to recover general weights. Nested anatomy (`face` $\subset$ `head` $\subset$ `body`) lets existing GT boxes act as hard geometric verifiers. | `ref:98cad413-b633-45d3-844f-f85b6e422447` |
| **4-parameter linear box-regression adapter for `face` pseudo-labels on `AnimeHeads`** | External Danbooru face detectors (`fuyucchi`, `deepghs_face`) include forehead/bangs (`mAP@75` drops to `0.026–0.088` on `Manga109` vs. Catnip's `0.644`). Fitting a linear $(\Delta x_c, \Delta y_c, s_w, s_h)$ adapter on the 24,785 `Manga109`/`izutsumi` `(pred_face, gt_head) -> gt_face` pairs aligns synthetic faces to tight `<face>` bounds. | `ref:98cad413-b633-45d3-844f-f85b6e422447` |
| **Retain `0: body` & generate `AnimeHeads` body pseudo-labels via `deepghs/anime_person_detection`** | `body` is essential for Izutsumi back/silhouette panels, Stage 1 hierarchical WBF filtering, and Stage 2 *Same-Panel Cannot-Link* constraints. Downsampling disembodied headshot crops in `AnimeHeads` yields 100% 3-class (`body`, `head`, `face`) slices and balances the 73k `AnimeHeads` vs. 10k `Manga109` ratio. | `ref:98cad413-b633-45d3-844f-f85b6e422447` |
| **Upgrade Stage 1 backbone from `yolo26n` (2.5M) to `yolo26m` (~20M) on Colab Pro L4** | `yolo26n` was constrained by the 16 GB M3 MacBook Air memory leak. Colab Pro L4 (24 GB VRAM) runs `yolo26m` at `imgsz=640`, `batch=48–64`, `amp=True` (~15–19 GB VRAM, ~6–7.5 min/epoch, ~18–21 ms inference latency) with 8× capacity to suppress complex manga background/screentone false positives. | `ref:98cad413-b633-45d3-844f-f85b6e422447` |
| **Color-to-line-art bridging (`RandomGrayscale` + HSV jitter)** | `AnimeHeads` (~73k images) is already RGB, but `hsv_h=0.0, hsv_s=0.0` in `config/pipeline.yaml` and domain-segregated labels taught a color shortcut. Enabling `hsv_h=0.015, hsv_s=0.5, hsv_v=0.4` + `RandomGrayscale(p=0.25)` forces shared contour features across B&W manga and colored fanart. | `ref:98cad413-b633-45d3-844f-f85b6e422447` |
| **Dynamic panel vs. page cropping + WBF + hierarchical containment in `SAHIExtractor`** | Pre-cropped *Dungeon Meshi* panels ($\max(H, W) \le 800\text{px}$) suffer scale distortion if sliced into `640×640` windows; gating SAHI by resolution + applying WBF and `face` $\subset$ `head` $\subset$ `body` containment pushes precision to **97–98%+**. | `ref:98cad413-b633-45d3-844f-f85b6e422447` |
| **Replace Stage 2 binary `ResNet18` Triplet Loss with Semi-Supervised Multi-Prototype Clustering** | `"not_izutsumi"` comprises dozens of distinct characters, and Izutsumi is strongly multi-modal *(standard cat-girl, feral/chibi reactions, hooded Ch. 31–33, body/tail silhouettes, colored fanart)*. Domain-pretrained embeddings (`CCIP` / `DINOv2` + `GeM`) + `FAISS` k-NN + `HDBSCAN` multi-prototype sub-clusters with panel/page constraints solve both problems. | `ref:98cad413-b633-45d3-844f-f85b6e422447` |

---

## Phase 1: Benchmark & Architecture Investigation [COMPLETE]
- [x] 1.1 Inspect `results/benchmarks/{face,head}/visuals/*.jpg` and `notebooks/catnipBenchmarkPresentation.ipynb` with Gemini Flash to diagnose missing Catnip Stage 1 face gallery overlays → `ref:b6f94a41-07e0-4ef3-9467-e193815270e8`
- [x] 1.2 Audit Stage 1 training configs, SAHI pre-slicing parity, dataset unification class boundaries (`izutsumi`, `manga109`, `ah_yolo`, `ah_coco`), and Stage 2 scaffolds → `ref:98cad413-b633-45d3-844f-f85b6e422447`
- [x] 1.3 Update `AGENTS.md` and `docs/architecture.md` with Stage 1 v2 (`yolo26m` + synthetic pseudo-labeling) and Stage 2 Semi-Supervised Clustering specifications → `ref:98cad413-b633-45d3-844f-f85b6e422447`

## Phase 2: Benchmark Artifact Refresh & Stage 1 Inference Upgrade (97–98% Precision) [CODE COMPLETE — 2.1 AWAITING REMOTE RUN]
- [ ] **2.1 Re-run full CUDA comparative benchmark (`scripts/colab/launch_eval_360.sh` / `scripts/eval/benchmark_faces.py`) and re-execute `notebooks/catnipBenchmarkPresentation.ipynb` so `face/visuals/` and `anime_illustration` domain tables reflect commit `9f0ff6b` fixes** ← OPERATIONAL: launch from 360 WSL; code path unchanged and green
- [x] 2.2 Implement dynamic panel-vs-page resolution gating (`panel_max_dim: 800`) and canonical page scale normalization in `src/detection/sahi_extractor.py` and `src/config.py` *(code complete: `tests/test_sahi_extractor.py`, predictor-seam pipeline; on-data precision validation pending the yolo26m deliverable)*
- [x] 2.3 Implement Weighted Box Fusion (WBF), per-class calibrated confidence thresholds (`conf_body`, `conf_head`, `conf_face`), and hierarchical anatomical containment filtering (`face` $\subset$ `head` $\subset$ `body`) in `src/detection/sahi_extractor.py` *(code complete: in-repo WBF — sahi 0.11.36 ships no WBF; face band exemption semantics per spec §2, regression-pinned)*

## Phase 3: Cross-Dataset Synthetic Pseudo-Labeling Pipeline (`scripts/unify/pseudo_label.py`) [CODE COMPLETE — 3.4 AWAITING DATA RUNS]
- [x] 3.1 Add teacher model catalog entries (`deepghs_head_x`, `deepghs_face_s`, `deepghs_person_s`, `zymk9_face`) and downloader support in `src/eval/adapters.py` *(code complete: catalog + `get_detector` target validation incl. "body")*
- [x] 3.2 Implement `src/preprocess/box_calibration.py` to fit the 4-parameter linear box-regression adapter $(\Delta x_c, \Delta y_c, s_w, s_h)$ on `Manga109` + `izutsumi` `(pred_face, gt_head) -> gt_face` pairs *(complete: Huber IRLS fit + validation quality floor + `--target-section calibrate` driver. **2026-11-06: fitted on real data locally (MPS)** — 10,884 pairs from 1,754 pages, `FaceBoxAdapter(α_x=-0.0096, α_y=+0.0238, β_w=0.9433, β_h=0.9313)`, validation recall@IoU0.75 0.455 → 0.523; serialized to `config/face_box_adapter.json`)*
- [x] 3.3 Implement `scripts/unify/pseudo_label.py` to generate geometrically verified `head` labels on `data/staging/manga109` and `body` + calibrated `face` labels on `data/staging/{ah_yolo,ah_coco}`, filtering out disembodied headshot-only crops without torsos *(code complete: Rule A/B + IoS verification, enriched output tree — inputs never mutated, rerun-converging writer, deterministic 10% headshot draw)*
- [x] 3.4 Re-run `pixi run unify-stage1` (`scripts/unify/stage1.py --slice`) on the fully 3-class annotated staging directories and archive `compressed/stage1_v2_sliced.tar` to Cloudflare R2 (`s3://catnip-canonical/`) *(**COMPLETE 2026-11-06, local MPS run; UPDATED 2026-11-07 with Izutsumi head/face remap + AnimeHeads torso geometry**):
  - Izutsumi Label Remap: Original 242 `class 2: face` remapped to `class 1: head` (median area 0.053); 204 tight `class 2: face` synthesized via `FaceBoxAdapter` + `deepghs_face_s`/`zymk9_face` ensemble.
  - AnimeHeads Torso Guard: Pruned whole-canvas boxes and enforced downward torso extension below chin (`torso_ext >= 0.35 * head_h`, `body_h >= 1.35 * head_h`). Correctly identified and downsampled 19,678 headshot crops.
  - Final Unified + Sliced Count: **54,015 images** → **135,213 slices** (`train`: 108,207, `val`: 13,481, `test`: 13,525), **1,340,555 annotations** across `body`, `head`, and `face`.
  - Unified Archive: 12.3 GB `compressed/stage1_v2_sliced.tar` uploaded to R2 via `rclone`.
  - Ground Truth Gallery: 40-sample interactive gallery (32 slices + 8 full pages) generated and published in `docs/gt_gallery/` (`pixi run visualize-gt`).)*

## Phase 4: Stage 1 v2 Training on Colab Pro L4 (`yolo26m` + Color Bridging) [IN PROGRESS]
- [x] 4.1 Update `config/pipeline.yaml` and `scripts/train/stage1.py` for `yolo26m.pt` (`batch=48`, `imgsz=640`, `cls_pw=1.0`, `hsv_h=0.015`, `hsv_s=0.5`, `hsv_v=0.4`, `close_mosaic=12`) and inject `RandomGrayscale(p=0.25)` + scan-contrast jitter into the Ultralytics training transform pipeline *(code complete: `YOLODataset.build_transforms` wrap, process-local patched-once; golden polish `--polish` with 3× izutsumi oversampled dataset; v1 M3 profile fully pinned on the pixi task; remote runner v2 env knobs + two-phase flow)*
- [🔄] 4.2 Execute Phase 1 training (75 epochs on full 3-class sliced dataset) on Colab Pro L4 via `scripts/colab/launch_360.sh`, delivering `models/yolo26m_stage1_v2.pt` directly from Phase 1 `best.pt` ← **IN PROGRESS (2026-11-10)**:
  - **Architectural Decision (2026-11-10):** Phase 2 golden polish (Izutsumi oversampling) has been **disabled** (`CATNIP_POLISH=0`). Stage 1 is strictly a class-agnostic anatomical detector (`0: body`, `1: head`, `2: face`). Oversampling Izutsumi risks art-style/character bias and weakens Stage 2's Same-Panel Cannot-Link negative constraints (Stage 1 must detect all characters equally).
  - Launched under autonomous 24/7 self-healing supervisor (`/root/catnip/supervisor.sh`) on `rifu360`. Automatically detects stale logs/disconnected Colab runtimes, clears hung wrappers, and resumes from R2 `runs/stage1_v2-last.pt`.
  - Current Status: **Epoch 67/75 (closing mosaic active)**. Throughput: **2.4 it/s** on Colab L4 (13.5 GB VRAM).
  - Validation Progress (13,481 val slices / 135,284 instances, all 3 classes):
    - Epoch 1: mAP50 0.852, mAP50-95 0.563
    - Epoch 3: mAP50 0.899, mAP50-95 0.635
    - Epoch 11: mAP50 0.931, mAP50-95 0.696
    - Epoch 25: mAP50 0.937, mAP50-95 0.710, Precision 0.882, Recall 0.890
    - Epoch 43: mAP50 0.942, mAP50-95 0.723, Precision 0.888, Recall 0.895
    - Epoch 66: **mAP50 0.944, mAP50-95 0.729, Precision 0.891, Recall 0.899**
  - Deliverable: Upon completion of Epoch 75, Phase 1 `best.pt` ships directly as `models/yolo26m_stage1_v2.pt` to R2.
  - Checkpoint: Rolling checkpoint pushed to `s3://catnip-canonical/runs/stage1_v2-last.pt`.
- [ ] 4.3 Benchmark `yolo26m_stage1_v2.pt` on the 16k test split and held-out *Dungeon Meshi* manga panels + colored fanart to verify $\ge 97\text{–}98\%$ precision across all 3 classes and both domains ← OPERATIONAL (awaiting completion of 4.2)

## Phase 5: Stage 2 Semi-Supervised Multi-Prototype Clustering (`src/reid/`) [CODE COMPLETE — TRAINING PENDING]
- [x] 5.1 Implement `src/reid/model.py` with a domain-pretrained character embedding backbone (`deepghs/ccip` or `DINOv2-ViT-S/14` / `ConvNeXt-Small` + `GeM` pooling $\rightarrow$ L2-normalized embedding) and multi-granularity instance fusion (`[face, head, body]`) *(code complete: ConvNeXt default + DINOv2 lazy hub; CCIP explicitly `NotImplementedError`; fusion renormalized over present parts)*
- [x] 5.2 Implement `src/reid/loss.py` and `scripts/train/stage2.py` using Supervised Contrastive Loss (`SupConLoss`) / Sub-Center ArcFace on Colab Pro L4 (preserving per-character cluster structure rather than collapsing `not_izutsumi` into one class) *(code complete: SupCon with positive-less-anchor guard, SubCenter ArcFace K=5, HDBSCAN pseudo-cluster mining with loud dense-budget cap; weight training itself runs on Colab)*
- [x] 5.3 Implement `src/reid/clustering.py` (`FAISS IndexFlatIP` k-NN graph + `HDBSCAN` density sub-clustering + Must-Link / Cannot-Link label propagation + *Same-Panel Cannot-Link* and *Same-Page Prior Boost* constraints) *(code complete: exact-numpy fallback for faiss, cannot-link veto, pre-boost-score panel ranking — a page-prior-promoted lookalike can never displace a natural core)*
- [x] 5.4 Wire `catnip extract`, `catnip embed`, `catnip match`, and `catnip pipeline` in `src/cli.py` with two-tier output ($\tau_{\text{core}}$ auto-tagging at $\ge 98\%$ precision + boundary cluster review export) *(code complete: embed-model provenance (`embed_meta.json`) pins match to the same embedding space; boundary review HTML gallery; bboxes flow through records for panel distinctness)*

---

# Detailed Technical Handoff Specifications

## 1. Benchmark Refresh (Task 2.1)

### Problem Summary
The committed `results/benchmarks/{face,head}/{metrics.json,summary.md,visuals/*.jpg}` were generated on `2026-11-05 15:50` before commit `9f0ff6b` (`17:44`), which fixed two bugs in [`scripts/eval/benchmark_faces.py`](../scripts/eval/benchmark_faces.py):
1. `classify_domain()` previously checked only `"ah_"` or `"anime"`, misclassifying 7,295 of 7,297 `AnimeHeads` slices (`ahv1_`, `ahv2_`, `ahv3_`, `ahaf_`) as `"general"` instead of `"anime_illustration"`.
2. Visual candidate ranking previously sorted all slices by `np.std(pred_counts)` without filtering `len(gt_boxes) > 0`, filling `results/benchmarks/face/visuals/` exclusively with `ahv1_`/`ahv2_` slices where `Ground Truth (0)` and `Catnip-Stage1 (0)`.

### Execution Commands
```bash
# On 360 WSL / Colab L4 GPU (full 15,993 test split -> syncs to R2):
bash scripts/colab/launch_eval_360.sh

# Pull updated results locally and re-render presentation notebook:
pixi run benchmark-presentation
pixi run python scripts/validate_notebooks.py notebooks/catnipBenchmarkPresentation.ipynb
```

---

## 2. Dynamic Panel/Page Inference, WBF & Hierarchical Filtering (Tasks 2.2 & 2.3)

### Target Files
- [`config/pipeline.yaml`](../config/pipeline.yaml)
- [`src/config.py`](../src/config.py)
- [`src/detection/sahi_extractor.py`](../src/detection/sahi_extractor.py)

### Specification
1. **Dynamic Panel vs. Full-Page Resolution Gate (`src/detection/sahi_extractor.py`):**
   - Inspect input image dimensions $(H, W)$ at the top of `SAHIExtractor.extract_from_image()` (Law 1 — Early Exit):
     - **Case A — Pre-Cropped Manga Panel ($\max(H, W) \le 800\text{px}$):** Bypass `get_sliced_prediction()`. Run direct single-pass Ultralytics inference at `imgsz=640` (letterboxed) so individual *Dungeon Meshi* panels are not chopped into partial sub-windows.
     - **Case B — High-Res / Variable-DPI Full Manga Page ($\max(H, W) > 800\text{px}$):** Normalize the image so the longer dimension $\max(H, W)$ is rescaled to a canonical page height (`canonical_page_max_dim = 1800` px) before running `640×640` SAHI slicing (`overlap_ratio = 0.2`), then project coordinates back to original pixel space. This guarantees identical character-to-slice scale parity across tankobon rips, magazine scans, and webrips.
2. **Per-Class Confidence & Hierarchical Anatomy Filter (Reaching 97–98%+ Precision):**
   - Replace the single global `confidence_threshold: 0.3` with per-class thresholds in `config/pipeline.yaml`:
     ```yaml
     sahi:
       slice_height: 640
       slice_width: 640
       overlap_ratio: 0.2
       perform_standard_pred: true
       postprocess_type: "WBF"        # Weighted Box Fusion (or NMM with IOS=0.40)
       postprocess_match_metric: "IOS"
       postprocess_match_threshold: 0.40
       panel_max_dim: 800             # <= 800px uses direct 640px inference
       canonical_page_max_dim: 1800   # normalizes full pages before slicing
       class_conf_thresholds:
         body: 0.52
         head: 0.52
         face: 0.56
       hierarchical_verify: true
     ```
   - **Hierarchical Containment Rule (`_filter_anatomical_hierarchy`):**
     - A `face` (`class 2`) detection with confidence $c \in [0.45, 0.70)$ is retained **if and only if** its Intersection-over-Self ($\text{IoS} = \frac{\text{Area}(B_{\text{face}} \cap B_{\text{parent}})}{\text{Area}(B_{\text{face}})}$) with at least one detected `head` (`class 1`) or `body` (`class 0`) box on the same image satisfies $\text{IoS} \ge 0.75$.
     - A `face` detection with $c \ge 0.70$ is unconditionally retained (handles extreme close-up panels where only a character's eyes/face fill the entire panel).

---

## 3. Cross-Dataset Synthetic Pseudo-Labeling Pipeline (Phase 3)

### Target Files
- `src/preprocess/box_calibration.py` *(new)*
- `scripts/unify/pseudo_label.py` *(new)*
- [`scripts/unify/stage1.py`](../scripts/unify/stage1.py)

### Teacher Models (Run Offline on Colab L4 GPU)
> **2026-11-06 correction:** the originally listed `face_detect_v1.4_x` and `person_detect_v1.2_x` URLs return 404 — deepghs publishes no x-sized face/person checkpoints (verified via the HF repo trees). The catalog (`src/eval/adapters.py`) uses the largest existing variants below.

| Target Class | Target Staging Dir | Teacher Checkpoint(s) | HuggingFace / Source URL |
|---|---|---|---|
| **`1: head`** | `data/staging/manga109` | `deepghs_head_x` (`YOLO11x`) + `Catnip-Stage1` | `https://huggingface.co/deepghs/anime_head_detection/resolve/main/head_detect_v2.0_x_yv11/model.pt` |
| **`2: face`** | `data/staging/{ah_yolo,ah_coco}` | `deepghs_face_s` (`YOLOv8s`) + `zymk9_face` | `https://huggingface.co/deepghs/anime_face_detection/resolve/main/face_detect_v1.4_s/model.pt` |
| **`0: body`** | `data/staging/{ah_yolo,ah_coco}` | `deepghs_person_s` (`YOLOv8s`) | `https://huggingface.co/deepghs/anime_person_detection/resolve/main/person_detect_v1.2_s/model.pt` |

### Step-by-Step Algorithmic Contract

#### Step 3.1: Fit the 4-Parameter Face Box Convention Adapter (`src/preprocess/box_calibration.py`)
External Danbooru face detectors (`deepghs_face_s`, `zymk9_face`) include forehead and bangs, whereas `Manga109` and `izutsumi` annotate tight brow-to-chin `<face>` boxes.
1. Run `deepghs_face_s` (WBF-fused with `zymk9_face`) on `data/staging/izutsumi` and a 2,000-page sample of `data/staging/manga109`.
2. Pair each predicted face $B_{\text{pred}} = (x_c^p, y_c^p, w^p, h^p)$ with its matched ground-truth tight face $B_{\text{gt}} = (x_c^g, y_c^g, w^g, h^g)$ where $\text{IoU} \ge 0.45$.
3. Fit the 4-parameter affine transformation via Huber/Ridge regression (robust to outliers):
   $$\hat{x}_c = x_c^p + \alpha_x w^p, \quad \hat{y}_c = y_c^p + \alpha_y h^p, \quad \hat{w} = \beta_w w^p, \quad \hat{h} = \beta_h h^p$$
4. Serialize $(\alpha_x, \alpha_y, \beta_w, \beta_h)$ to `config/face_box_adapter.json` and assert on the validation split that transforming `deepghs_face_s` predictions raises `Manga109` `mAP@75` from `<0.10` to `>0.45`.

#### Step 3.2: Generate Verified `head` (`class 1`) Labels on `Manga109`
For each image and label file in `data/staging/manga109` (which currently contains only `0: body` and `2: face`):
1. Run `deepghs_head_x` (`conf >= 0.45`) + `Catnip-Stage1` (`conf >= 0.45`) and fuse predictions via WBF.
2. Accept a candidate `head` box $H$ **if and only if**:
   - **Rule A (Face Enclosure):** $H$ encloses a ground-truth `<face>` box $F$ with $\text{IoS}(F, H) \ge 0.80$ and area ratio $1.15 \le \frac{\text{Area}(H)}{\text{Area}(F)} \le 4.5$ (clamping $H$ so it fully covers $F$: $H_{x1} = \min(H_{x1}, F_{x1})$, etc.); **OR**
   - **Rule B (Back-of-Head / No-Face Body):** For a ground-truth `<body>` box $B$ that has no associated `<face>`, $H$ lies within the top 35% of $B$ with $\text{IoS}(H, B) \ge 0.75$ and confidence $\ge 0.65$.
3. Append accepted `1 <xc> <yc> <w> <h>` lines to the staging `.txt` label file.

#### Step 3.3: Generate Verified `body` (`class 0`) & Calibrated `face` (`class 2`) Labels on `AnimeHeads`
For each image and label file in `data/staging/ah_yolo` and `data/staging/ah_coco` (which currently contain only `1: head`):
1. **Calibrated `face` (`class 2`):**
   - Run `deepghs_face_s` + `zymk9_face` (`conf >= 0.45`).
   - Keep only face predictions strictly inside a ground-truth `head` box $H_{\text{gt}}$ ($\text{IoS}(F_{\text{pred}}, H_{\text{gt}}) \ge 0.85$, at most 1 face per GT head).
   - Apply the `face_box_adapter.json` shrinkage transform $(\alpha_x, \alpha_y, \beta_w, \beta_h)$ and write as `2 <xc> <yc> <w> <h>`.
2. **Verified `body` (`class 0`) & Headshot Filtering:**
   - Run `deepghs_person_s` (`conf >= 0.50`).
   - Keep predicted `body` boxes $B_{\text{pred}}$ that enclose at least one GT `head` box $H_{\text{gt}}$ ($\text{IoS}(H_{\text{gt}}, B_{\text{pred}}) \ge 0.70$) and actually include a torso ($\text{IoU}(B_{\text{pred}}, H_{\text{gt}}) \le 0.80$, i.e. the body box is significantly larger than just the head).
   - **Dataset Balancing Filter:** If an `AnimeHeads` image has **0 valid `body` boxes** (meaning it is a tightly cropped disembodied avatar/headshot with no torso), **exclude or downsample it by 90%** (`keep_prob = 0.10`). This trims `AnimeHeads` from ~73k raw images down to ~28k–35k **complete 3-class (`body`, `head`, `face`) illustrations**, balancing the ratio against `Manga109` (~10k pages) and eliminating partial-label gradient contradiction.

---

## 4. Stage 1 v2 Training on Colab Pro L4 (`yolo26m` + Color Bridging) (Phase 4)

### Target Files
- [`config/pipeline.yaml`](../config/pipeline.yaml)
- [`scripts/train/stage1.py`](../scripts/train/stage1.py)
- [`scripts/colab/remote_runner.py`](../scripts/colab/remote_runner.py)

### L4 Training Hyperparameters (`config/pipeline.yaml`)
```yaml
training:
  stage1:
    model: yolo26m.pt
    epochs: 75
    patience: 15
    batch: 48                  # ~15.5 GB VRAM on L4 (24 GB); OOM auto-fallback to 32
    imgsz: 640
    device: cuda
    optimizer: MuSGD
    lr0: 0.001
    cos_lr: true
    cls_pw: 1.0                # restored from 0.6 now that all 3 classes are fully labeled
    box: 8.5
    dfl: 2.0
    weight_decay: 0.002
    freeze: 0
    workers: 8
    use_subset: false
    mosaic: 0.5
    close_mosaic: 12
    mixup: 0.10
    hsv_h: 0.015               # re-enabled for colored artwork/fanart generalization
    hsv_s: 0.50
    hsv_v: 0.40
    degrees: 2.0
```

### Custom Color-to-Line-Art Bridge Callback (`scripts/train/stage1.py`)
Register an Ultralytics custom Albumentations/torchvision transform in `scripts/train/stage1.py`:
- `RandomGrayscale(p=0.25)`: Converts 25% of colored `AnimeHeads` training slices to 3-channel grayscale (with random contrast/screentone grain), forcing `yolo26m` early conv filters to learn **shared structural line/contour features** across B&W manga and colored fanart.
- **Phase 2 Golden Polish (10 epochs):** After the 75-epoch run converges, run 10 epochs at `lr0=0.0001`, `mosaic=0.0` with `izutsumi` slices oversampled `3×` so final bounding-box regression is calibrated to *Dungeon Meshi* annotations.

---

## 5. Stage 2: Semi-Supervised Multi-Prototype Clustering (Phase 5)

### Target Module Structure (`src/reid/`)
```
src/reid/
├── __init__.py
├── model.py          # Domain-pretrained backbone (CCIP / DINOv2 / ConvNeXt) + GeM + Multi-Granularity Fusion
├── dataset.py        # CharacterInstanceDataset (links nested [face, head, body] crops + panel/page metadata)
├── loss.py           # SupConLoss (Supervised Contrastive Loss) + SubCenterArcFace
└── clustering.py     # FAISS IndexFlatIP k-NN graph + HDBSCAN sub-clusters + Constrained Label Propagation
```

### Architectural Contract

#### 5.1 Embedding Backbone & Multi-Granularity Instance Fusion (`src/reid/model.py`)
1. **Backbone:** Support pluggable anime/manga-pretrained backbones (default: `deepghs/ccip` or `timm` `vit_small_patch14_dinov2` / `convnext_small.fb_in22k` with **Generalized Mean (`GeM`) Pooling**, $p=3.0$ learnable, projecting to a **256-d L2-normalized embedding**).
2. **Multi-Granularity Crop Linking:**
   - On each image/panel, group Stage 1 detections into **Character Instances** by hierarchical containment ($F \subset H \subset B$).
   - Compute individual L2-normalized embeddings $(\mathbf{e}_{\text{face}}, \mathbf{e}_{\text{head}}, \mathbf{e}_{\text{body}})$ and fuse available parts via confidence-weighted projection:
     $$\mathbf{e}_{\text{instance}} = \text{L2Norm}\left( w_f c_f \mathbf{e}_{\text{face}} + w_h c_h \mathbf{e}_{\text{head}} + w_b c_b \mathbf{e}_{\text{body}} \right)$$
     *(Default weights: $w_h = 0.50, w_f = 0.35, w_b = 0.15$, dynamically renormalized over present parts so back-of-body or head-only crops still produce valid embeddings in the shared space).*

#### 5.2 Contrastive Fine-Tuning Without the Heterogeneous Negative Trap (`src/reid/loss.py`)
- Instead of binary Triplet Loss that pulls all `"not_izutsumi"` crops together:
  1. Run offline **HDBSCAN** over initial pretrained embeddings of `"not_izutsumi"` crops so recurring non-Izutsumi characters (Laios, Marcille, Chilchuck, Senshi) receive distinct pseudo-cluster IDs $y \in \{1, \dots, C\}$, while singleton background NPCs are treated as pure repulsion negatives.
  2. Fine-tune on Colab L4 using **Supervised Contrastive Loss (`SupConLoss`, $\tau = 0.07$)** + **Sub-Center ArcFace ($K=5$ sub-centers per class)** so Izutsumi's distinct visual modes are explicitly permitted to occupy separate sub-centers without penalty.

#### 5.3 FAISS + HDBSCAN Multi-Prototype Clustering & Contextual Constraints (`src/reid/clustering.py`)
1. **Multi-Prototype Sub-Clusters:**
   - Build a `FAISS IndexFlatIP` index over all extracted character instances.
   - Partition the k-NN similarity graph ($k=25$) using **HDBSCAN** (`min_cluster_size=5`, `metric='precomputed'` cosine distance).
   - Seed with labeled Label Studio anchors:
     - **Must-Link Positive Seeds:** `izutsumi_{face,head,body}` crops define verified **Izutsumi Sub-Cluster Medoids** $\{\mathbf{m}_1, \dots, \mathbf{m}_M\}$ *(naturally representing: (1) standard B&W cat-girl face, (2) feral/gremlin chibi reactions, (3) hooded Ch. 31–33 appearances, (4) full-body tail/silhouette views, and (5) colored fanart)*.
     - **Cannot-Link Negative Seeds:** `other_{face,head,body}` crops mark non-Izutsumi character clusters (Laios, Marcille, Senshi, etc.).
2. **Contextual Panel & Page Co-Occurrence Rules:**
   - **Rule 1 — Same-Panel Cannot-Link (Mutual Exclusion):** If a single manga panel contains $N \ge 2$ spatially distinct character instances ($I_a, I_b$ with $\text{IoU}(B_a, B_b) < 0.15$), at most **one** instance in that panel can be assigned to `izutsumi` (the highest-scoring instance wins; the runner-up is suppressed unless manual override is set for doppelganger chapters).
   - **Rule 2 — Same-Page Prior Boost:** If page $P$ contains a high-confidence core Izutsumi detection ($\max_k \langle \mathbf{e}, \mathbf{m}_k \rangle \ge \tau_{\text{core}} = 0.82$) in Panel $i$, boost the similarity score of candidate partial/back/body instances in adjacent panels of page $P$ by $+\Delta_{\text{page}} = 0.06$ (provided Rule 1 is respected).
3. **Two-Tier Decision Output (`src/cli.py`):**
   - **Tier 1 — Auto-Tagged Core ($\text{score} \ge \tau_{\text{core}} = 0.82$):** Automatically tagged as `izutsumi` (**target precision $\ge 98\%$**).
   - **Tier 2 — Boundary Sub-Clusters ($\text{score} \in [0.68, 0.82)$):** Grouped by HDBSCAN sub-cluster ID into an HTML/Notebook review gallery so the user can accept or reject an entire visual sub-cluster (e.g., a new outfit or fanart style) in a single action.

---

## Notes
- 2026-11-06: Visual inspection via Gemini Flash confirmed `Catnip-Stage1` renders cyan overlays properly on `head/visuals/compare_ahaf_*.jpg`, and that `face/visuals/` lacked overlays only because the 15:50 run predated commit `9f0ff6b` `ref:b6f94a41-07e0-4ef3-9467-e193815270e8`
- 2026-11-06: Codebase audit confirmed `AnimeHeads` (~73k images) is preserved in full RGB during slicing, meaning color generalization only requires cross-labeling `face`/`body` onto `AnimeHeads` and enabling HSV + `RandomGrayscale` bridging `ref:98cad413-b633-45d3-844f-f85b6e422447`
- 2026-11-06: Open-source naming (`Catseye` / `Tapetum` + `Clowder` vs. `Koma` vs. `blorbo`) recorded in `AGENTS.md` as **Pending Decision**.
- 2026-11-06: Code complete for tasks 2.2/2.3, 3.1–3.3, 4.1, 5.1–5.4 (GLM-5.3 parallel build, Mistral Large 4 review, Sonnet 5.5 final review; 264 synthetic tests green). Review-settled semantics: the plan's face hierarchy rule `[0.45, 0.70) iff contained / >= 0.70 unconditional` governs faces when `hierarchical_verify` is on (the `face: 0.56` class threshold applies only when the hierarchy is disabled — YAML documented); Same-Panel Cannot-Link ranks by pre-boost score so page-prior-promoted lookalikes never displace natural cores; Rule B enforces the full head box inside the top-35% band region; faiss/timm/hdbscan remain optional at runtime (exact numpy kNN + sklearn HDBSCAN defaults), so no new hard dependencies were added.
- 2026-11-06: New flow: pseudo-label `--target-section calibrate` fits + validates the face box adapter end-to-end (teacher runs → IoU-0.45 pair matching → Huber fit → held-out validation floor → `config/face_box_adapter.json`). Operational tasks (2.1 benchmark re-run, 3.4 unify+archive, 4.2 L4 training, 4.3 v2 benchmark) still require the 360 WSL / Colab L4 runners and mounted data.
- 2026-11-06 (session close): **task 3.4 delivered** — the full enrichment/re-slice/archive chain ran locally on MPS (see 3.4 for the numbers) and the 13 GB v2 dataset is published in R2 as 46 byte-verified parts. `origin/main` pushed with the fixed harness (2.1 unblocked) and all v2 code; `models/yolo26m.pt` uploaded (md5-verified). Launchers audited for v2: both runners clone `CATNIP_REPO_BRANCH` (default main), `launch_360.sh` forwards the `CATNIP_STAGE1_*`/`CATNIP_POLISH*` profile, `launch_eval_360.sh` gains `CATNIP_EVAL_MODEL_KEY`/`CATNIP_EVAL_RESULTS_PREFIX` for the 4.3 variant. Remaining remote-only: 2.1 and 4.2/4.3 from the 360 WSL host; Stage 2 seed crops (Label Studio `stage2/{izutsumi,not_izutsumi}`) are still the open user-side prerequisite for Phase 5 training.
