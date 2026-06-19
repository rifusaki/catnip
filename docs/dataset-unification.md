# Dataset Unification Pipeline (Stage 1)

> **Audience:** Someone who has cloned the repo and needs to generate the Stage 1 training dataset from source annotations.

The dataset unification pipeline converts six distinct source datasets into a single, shuffled, and split 3-class YOLO training dataset. The output is consumed directly by YOLO26 for Stage 1 localization training (detect bodies, heads, and faces regardless of character identity).

## Overview

The pipeline runs in two phases:

| Phase | Step | Input | Output |
|-------|------|-------|--------|
| **Convert** | Scripts 1–4 | Source annotations in native format (JSON, XML, YOLOv8 txt, COCO JSON) | Flat YOLO format in 4 staging directories |
| **Merge** | Script 5 | 4 staging directories | Shuffled 80/10/10 split with `dataset.yaml` and `split_manifest.json` |

All scripts live in `scripts/unify/` and are run as `python scripts/unify/<name>.py`. Every script supports `--help` for inline flag documentation. The actual parsing logic lives in the importable library modules under `src/preprocess/`.

### Class Mapping

The entire pipeline uses a single 3-class scheme:

| Class ID | Name | Sources |
|----------|------|---------|
| `0` | body | izutsumi, Manga109 |
| `1` | head | ah_yolo, ah_coco |
| `2` | face | izutsumi, Manga109 |

Manga109 provides no head annotations (both YOLO head datasets fill this gap). The head class is always added by remapping during conversion — no source dataset produces native class 1 labels.

### Data Flow

```
Source datasets                    Staging dirs                   Training
─────────────────                  ────────────                   ────────
izutsumi LS JSON   ──[1]──►  staging/izutsumi/
manga109 XML       ──[2]──►  staging/manga109/    ──                      ──
ah_detection v1    ──[3]──►  staging/ah_yolo/       ──[5]──►  training/stage1/
ah_detection v2    ──[3]──►  staging/ah_yolo/       ──        (80/10/10 split)
ani_face_detection ──[3]──►  staging/ah_yolo/       ──        dataset.yaml
AnimeHeadsv3 COCO  ──[4]──►  staging/ah_coco/    ──          split_manifest.json
```

```
[1] scripts/unify/izutsumi.py    → src/preprocess/convert_labels.py
[2] scripts/unify/manga109.py    → src/preprocess/manga109_parser.py
[3] scripts/unify/yolo_heads.py  → src/preprocess/yolo_remap.py
[4] scripts/unify/coco_heads.py  → src/preprocess/coco_parser.py
[5] scripts/unify/stage1.py      (standalone orchestration)
```

## Prerequisites

### Source Data

The following directories must exist under `catnip-data/data/` before running any conversion scripts:

| Directory | Contents | Required By |
|-----------|----------|-------------|
| `izutsumi/annotations/` | Per-annotation Label Studio JSON exports (76 files, chapters 31–106) | `scripts/unify/izutsumi.py` |
| `izutsumi/manga/` | Manga page images in `v01/` through `v14/` subdirectories | `scripts/unify/izutsumi.py` |
| `manga109/annotations/` | 109 per-title XML files | `scripts/unify/manga109.py` |
| `manga109/images/{title}/` | Per-title image subdirectories with `{index:03d}.jpg` page images | `scripts/unify/manga109.py` |
| `anime_head_detection/v1/` | YOLOv8 dataset with `train/`, `valid/`, `test/` splits | `scripts/unify/yolo_heads.py` |
| `anime_head_detection/v2/` | YOLOv8 dataset with `train/`, `valid/`, `test/` splits | `scripts/unify/yolo_heads.py` |
| `anime_head_detection/ani_face_detection/` | YOLOv8 dataset with `train/`, `valid/`, `test/` splits | `scripts/unify/yolo_heads.py` |
| `AnimeHeadsv3/train/_annotations.coco.json` | COCO-format annotations + images | `scripts/unify/coco_heads.py` |
| `AnimeHeadsv3/valid/_annotations.coco.json` | COCO-format annotations + images | `scripts/unify/coco_heads.py` |
| `AnimeHeadsv3/test/_annotations.coco.json` | COCO-format annotations + images | `scripts/unify/coco_heads.py` |

### Software

```bash
# Minimal — just the standard library for conversion scripts.
# scripts/unify/izutsumi.py reuses src/preprocess/convert_labels.py (also stdlib + json).
python >= 3.11

# For running tests:
pixi install
```

## Quick Start

Run the conversion scripts in order, then the merge script. Scripts 1–4 are independent and can run in parallel, but all four must complete before running script 5.

```bash
# Phase 1: Convert each source dataset to flat YOLO staging format

# 1. Izutsumi (Label Studio JSON → YOLO)
python scripts/unify/izutsumi.py

# 2. Manga109 (XML → YOLO)
python scripts/unify/manga109.py

# 3. Anime Head Detection YOLO datasets (class remap 0→1)
python scripts/unify/yolo_heads.py

# 4. AnimeHeadsv3 COCO (COCO → YOLO, class remap → 1)
python scripts/unify/coco_heads.py \
    --input-dir catnip-data/data/AnimeHeadsv3 \
    --output-dir catnip-data/data/staging/ah_coco

# Phase 2: Merge, shuffle, split, generate training-ready output
python scripts/unify/stage1.py
```

### Verifying Output

After running all five scripts, check:

```bash
# Staging directories (flat YOLO format, one .txt per image)
ls catnip-data/data/staging/izutsumi/images/  # izutsumi_v09_0043.jpg, ...
ls catnip-data/data/staging/izutsumi/labels/   # izutsumi_v09_0043.txt, ...
ls catnip-data/data/staging/manga109/images/   # manga109_*.jpg, ...
ls catnip-data/data/staging/ah_yolo/images/    # ahv1_*, ahv2_*, ahaf_*, ...
ls catnip-data/data/staging/ah_coco/images/    # ahv3_*.jpg, ...

# Final training output (nested YOLO format with splits)
tree catnip-data/training/stage1/
# Expected:
# catnip-data/training/stage1/
# ├── dataset.yaml
# ├── split_manifest.json
# ├── images/
# │   ├── train/
# │   ├── val/
# │   └── test/
# └── labels/
#     ├── train/
#     ├── val/
#     └── test/
```

### Dry Run

To preview the merge without writing files:

```bash
python scripts/unify/stage1.py --dry-run
```

This prints per-source contributions, per-split class distributions, and overall totals.

## Format Details

### Staging Format (Scripts 1–4 Output)

Flat directory structure — one `.txt` label file per image, both sharing the same stem:

```
staging/<source>/
├── images/
│   ├── izutsumi_v09_0043.jpg
│   ├── manga109_ISFCFK_069.jpg
│   └── ...
└── labels/
    ├── izutsumi_v09_0043.txt
    ├── manga109_ISFCFK_069.txt
    └── ...
```

### Training Format (Script 5 Output)

Nested directory structure — images and labels split into `train/`, `val/`, `test/`:

```
training/stage1/
├── dataset.yaml
├── split_manifest.json
├── images/
│   ├── train/
│   │   ├── izutsumi_v09_0043.jpg
│   │   └── ...
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    │   ├── izutsumi_v09_0043.txt
    │   └── ...
    ├── val/
    └── test/
```

### YOLO Label Format

Each line in a `.txt` label file represents one bounding box:

```
<class_id> <x_center> <y_center> <width> <height>
```

All four coordinates are normalized to `[0, 1]` relative to image dimensions. Example:

```
0 0.4523 0.6120 0.2340 0.0890
1 0.3045 0.2104 0.0567 0.0712
2 0.6721 0.3412 0.0443 0.0589
```

- `x_center`, `y_center`: center of the bounding box (not top-left)
- `width`, `height`: box dimensions
- `class_id`: 0 (body), 1 (head), or 2 (face)

## Script Reference

### 1. `scripts/unify/izutsumi.py`

**Purpose:** Convert izutsumi Label Studio per-annotation JSON exports to flat YOLO format.

**Input:**
- Annotation JSONs from `catnip-data/data/izutsumi/annotations/` (76 per-annotation files, chapters 31–106)
- Manga page images from `catnip-data/data/izutsumi/manga/` (volume subdirectories `v09/`, `v14/`)

**Output:** `catnip-data/data/staging/izutsumi/{images,labels}/` with prefix `izutsumi_`
- ~75 images
- ~385 annotations (body class 0 + face class 2)

**How it works:**
1. Calls `convert_annotations_directory` from `src/preprocess/convert_labels.py` to parse Label Studio JSON and produce YOLO-format labels in a temp directory. The converter groups per-annotation JSONs by source image and remaps fine-grained labels (`izutsumi_body` → body, `other_face` → face, etc.) to the 3-class Stage 1 scheme.
2. Walks the temp directory, renames each label file to a flat stem (e.g., `v09/0043.txt` → `izutsumi_v09_0043.txt`).
3. Copies the corresponding manga image with the renamed stem.

**Key flags:**

| Flag | Default | Purpose |
|------|---------|---------|
| `--input-dir` | `catnip-data/data/izutsumi/annotations` | Per-annotation JSON directory |
| `--manga-dir` | `catnip-data/data/izutsumi/manga` | Root of manga image tree (`v01/`–`v14/`) |
| `--output-dir` | `catnip-data/data/staging/izutsumi` | YOLO output directory |
| `--verbose` | off | Enable debug-level logging |

### 2. `scripts/unify/manga109.py`

**Purpose:** Convert Manga109 XML annotations to flat YOLO format.

**Input:**
- 109 XML files from `catnip-data/data/manga109/annotations/`
- Page images from `catnip-data/data/manga109/images/{title}/`

**Output:** `catnip-data/data/staging/manga109/{images,labels}/` with prefix `manga109_`
- ~10,000 images
- ~275,000 annotations (body class 0 + face class 2 — Manga109 has no head annotations)

**How it works:**
1. Parses each XML file with `xml.etree.ElementTree`, extracting `<body>` and `<face>` elements from `<pages>/<page>`.
2. Converts pixel coordinates (`xmin`, `ymin`, `xmax`, `ymax`) to YOLO normalized format using the page's `width` and `height` attributes.
3. Validates every bounding box: skips any box whose center falls outside [0,1], whose width or height exceeds 1.0, or whose dimensions are non-positive. These are tallied as OOB (out-of-bounds) skipped annotations.
4. Copies images and writes label `.txt` files with naming pattern `manga109_{title}_{index:03d}`.

**Key flags:**

| Flag | Default | Purpose |
|------|---------|---------|
| `--annotations-dir` | `catnip-data/data/manga109/annotations` | XML annotation file directory |
| `--images-dir` | `catnip-data/data/manga109/images` | Per-title image subdirectory root |
| `--output-dir` | `catnip-data/data/staging/manga109` | YOLO output directory |
| `--verbose` | off | Enable debug-level logging |

### 3. `scripts/unify/yolo_heads.py`

**Purpose:** Convert three anime head detection YOLOv8 datasets to unified flat YOLO format, remapping all source class 0 → target class 1 (head).

**Input:**
- `anime_head_detection/v1/` — YOLOv8 dataset with `train/`, `valid/`, `test/` splits
- `anime_head_detection/v2/` — YOLOv8 dataset with `train/`, `valid/`, `test/` splits
- `anime_head_detection/ani_face_detection/` — YOLOv8 dataset with `train/`, `valid/`, `test/` splits

**Output:** `catnip-data/data/staging/ah_yolo/{images,labels}/`
- ~65,000 images, all head class (1)
- Filename prefixes: `ahv1_`, `ahv2_`, `ahaf_`

**How it works:**
1. Processes all three dataset variants by default (customizable via `--datasets`).
2. For each split within each variant, iterates over label `.txt` files, parses each line, and remaps class 0 → 1.
3. If any line contains an unknown class ID (anything other than 0), the **entire label file is discarded** — this is a deliberate poison-on-first-error design.
4. Handles deduplication: if the same stem appears in multiple splits of the same dataset, the first occurrence wins and subsequent occurrences are skipped with a warning.
5. Copies images and writes remapped labels into the flat staging structure.

**Key flags:**

| Flag | Default | Purpose |
|------|---------|---------|
| `--datasets` | All 3 variants (`v1`, `v2`, `ani_face_detection`) | Subset of dataset paths to process |
| `--output-dir` | `catnip-data/data/staging/ah_yolo` | YOLO output directory |
| `--verbose` | off | Enable debug-level logging |

### 4. `scripts/unify/coco_heads.py`

**Purpose:** Convert the AnimeHeadsv3 COCO-format dataset to flat YOLO format.

**Input:**
- `AnimeHeadsv3/{train,valid,test}/_annotations.coco.json` (one per split)
- Image files in each split directory

**Output:** `catnip-data/data/staging/ah_coco/{images,labels}/` with prefix `ahv3_`
- ~8,000 images
- ~24,000 annotations, all head class (1)

**How it works:**
1. Validates that all three splits (`train/`, `valid/`, `test/`) exist and each contains `_annotations.coco.json`.
2. Builds an `image_id → (file_name, width, height)` map from the COCO `images` array.
3. Converts COCO pixel-format bboxes `[x, y, w, h]` to YOLO normalized `[cx, cy, w, h]` using each image's actual dimensions.
4. Handles partial overflow: bboxes that extend beyond image boundaries are clamped to [0, 1] and tracked in the summary as "clamped." Fully degenerate or entirely out-of-bounds bboxes are dropped.
5. Both COCO category IDs (0 and 1) map to YOLO class 1 (head) — AnimeHeadsv3 labels both as `"head"`.
6. Images with zero usable annotations after filtering are skipped (their copied image file is deleted).

**Key flags:**

| Flag | Default | Purpose |
|------|---------|---------|
| `--input-dir` | *(required)* | Root of AnimeHeadsv3 (contains `train/`, `valid/`, `test/`) |
| `--output-dir` | *(required)* | YOLO output directory |
| `--verbose` | off | Enable debug-level logging |

### 5. `scripts/unify/stage1.py`

**Purpose:** Merge all four staging directories, shuffle deterministically, split into train/val/test, and generate training-ready output.

**Input:**
- `staging/izutsumi/` (from script 1)
- `staging/manga109/` (from script 2)
- `staging/ah_yolo/` (from script 3)
- `staging/ah_coco/` (from script 4)

**Output:** `catnip-data/training/stage1/`
- `dataset.yaml` — absolute paths, 3-class names
- `split_manifest.json` — per-source per-split image counts
- `images/{train,val,test}/` and `labels/{train,val,test}/` — nested YOLO format

**Expected totals (full pipeline):**
- 83,494 images
- 550,807 annotations
- Class distribution: body 28.6%, head 49.9%, face 21.6%

**How it works:**
1. **Collect**: Walks all four staging directories, collecting `(image_path, label_path, source_name)` triples. Gracefully warns on missing or incomplete staging directories.
2. **Deduplicate**: If the same image stem appears in multiple staging directories, only the first occurrence is kept. Duplicates are tallied.
3. **Shuffle**: Deterministically shuffles all pairs using `random.Random(seed)`. Default seed is 42.
4. **Split**: Divides shuffled pairs into train/val/test by ratio (default 80/10/10). Fails early if any split would be empty.
5. **Copy**: Copies images and labels into the nested output structure. The output directory is cleaned before writing (previous contents are removed).
6. **Metadata**: Writes `dataset.yaml` (absolute paths for YOLO training) and `split_manifest.json` (per-source per-split image counts for auditability).

**Key flags:**

| Flag | Default | Purpose |
|------|---------|---------|
| `--split` | `80 10 10` | Train/val/test ratios (must sum to 100) |
| `--output-dir` | `catnip-data/training/stage1` | Final output directory |
| `--seed` | `42` | Random seed for reproducible shuffling |
| `--dry-run` | off | Preview stats only — no files written |
| `--verbose` | off | Enable debug-level logging |

## Configuration

The pipeline reads classification labels from `config/pipeline.yaml`:

```yaml
labels:
  stage1:
    body: 0
    head: 1
    face: 2
```

Path defaults in the conversion scripts are hardcoded relative to the project root and align with the paths defined in `config/pipeline.yaml` → `paths:`. The merge script (`scripts/unify/stage1.py`) resolves staging directories from constants at the top of the file.

Generated `dataset.yaml` uses **absolute paths** for compatibility with YOLO training:

```yaml
path: /absolute/path/to/catnip-data/training/stage1
train: images/train
val: images/val
test: images/test

names:
  0: body
  1: head
  2: face
```

## Related Modules

### `src/training/preparation.py`

Contains `split_data(imgs, val_ratio=0.2, seed=42)` for 80/20 splitting and `prepare_triplet_dataset(...)` for Stage 2 metric learning data preparation. The Stage 2 triplet dataset uses 2 classes (`izutsumi`/`not_izutsumi`) and is separate from the Stage 1 unification pipeline described in this document.

## Troubleshooting

### "No pairs found in any staging directory"

`scripts/unify/stage1.py` reports this when **none** of the four staging directories exist or contain usable image-label pairs. Run each conversion script (1–4) first and verify their output:

```bash
ls catnip-data/data/staging/izutsumi/images/ | head
ls catnip-data/data/staging/izutsumi/labels/  | head
# Repeat for manga109, ah_yolo, ah_coco
```

### "Staging dir not found — skipping"

Each conversion script must complete before `scripts/unify/stage1.py` can collect its output. If the merge script skips a staging directory, re-run the corresponding conversion script and check for errors in its summary.

### Duplicate stems

If the same filename stem appears in multiple staging directories, `scripts/unify/stage1.py` keeps the first occurrence and discards the rest. This is logged as a warning. Duplicate stems across different datasets are rare but possible (e.g., a manga page image integrated from two sources). To investigate which sources clash:

```bash
python scripts/unify/stage1.py --verbose 2>&1 | grep "Duplicate stem"
```

### Out-of-bounds bounding boxes (Manga109)

Manga109 XML occasionally contains annotations whose coordinates fall outside the declared page dimensions. `scripts/unify/manga109.py` skips these and tallies them as "skipped (OOB/deg)" in the summary. A small number of skipped boxes is normal. If the count is unexpectedly high, investigate the XML source.

### Clamped bounding boxes (COCO)

AnimeHeadsv3 COCO annotations sometimes exceed image boundaries (partial overflow). `scripts/unify/coco_heads.py` clamps these to [0, 1] and tracks the count in the summary under "bboxes clamped." This is expected behavior and typically affects a negligible fraction of annotations. Use `--verbose` to see per-annotation debug messages.

### Unknown class IDs (YOLO heads)

If `scripts/unify/yolo_heads.py` encounters a class ID other than 0 in any YOLO label file, it discards the **entire file** (not just the offending line). This is logged as "malformed." Check the source dataset labels if the malformed count is high.

### Empty split errors

If `scripts/unify/stage1.py` exits with "Split 'test' is empty" or similar, there are fewer than 10 image-label pairs (the minimum threshold). Either the conversion scripts produced insufficient data, or the split ratios are too extreme. Verify that staging directories contain data and adjust `--split` if needed.

### Cannot import `src.preprocess.convert_labels`

`scripts/unify/izutsumi.py` depends on this module. Run from the project root (not from inside `scripts/unify/`):

```bash
# Correct — from project root
python scripts/unify/izutsumi.py

# Incorrect — relative imports will fail
cd scripts/unify && python izutsumi.py
```
