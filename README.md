# Catnip
> an izutsumi identification project
> man i love izutsumi

Catnip is for those obsessive lovers of a specific manga character with spare time and computing power.

The goal of this project is to answer the age-old question of **"is Izutsumi in this image?"**

![izutsumi](https://data.izutsumi.art/manga/SmartSelect_20240620_141333_Gallery.webp)

Input some manga pages, add some seeds, and rejoice at the sight of matches. I haven't implemented an actual method to export the files. Yet.

---

## Architecture

I am using a two-stage pipeline: **detect** (localization) and **identify** (classification).

1. **Stage 1 (Localization):** Slices full manga pages into overlapping 640×640 patches (SAHI) and runs a fine-tuned **YOLO26 Nano** to find all bodies, heads, and faces regardless of character identity. Training data is pre-sliced to the exact same 640px dimensions for train/inference scale parity.
2. **Stage 2 (Identification):** Metric learning (ResNet18 + GeM pooling + Triplet Loss) extracts 128-d embeddings from the crops and compares them against Izutsumi seed references using FAISS cosine similarity.

---

## Usage

My greatest challenge has been getting some kind of viable GPU and RAM for this. So I've tried a few. I suggest using [Pixi](https://pixi.sh/latest/) for package management.

```bash
# Install dependencies
pixi install

# Prepare datasets + pre-slice for SAHI parity
pixi run data-prep
pixi run sanitize-names

# Train Stage 1
# Option A: M3 Mac Air (no)
pixi run subset-stage1
pixi run train-stage1-m3-chain

# Option B: "Headless" Colab
cd scripts/colab && ./launch_360.sh
```

---

## Datasets

- **izutsumi**: Own dataset. I used Label Studio to manually annotate bounding boxes on *Dungeon Meshi* manga pages. The labels allow for the usage of the same dataset in both stages.
- **[manga109](https://huggingface.co/datasets/hal-utokyo/Manga109)**: Double-page spreads parsed with `manga109api` to extract character bodies and faces.
- **[deepghs/anime_head_detection](https://huggingface.co/datasets/deepghs/anime_head_detection)**: High-quality head annotations (v1, v2, and ani_face_detection) remapped to fill the head class gap.
- **[nyuuzyou/AnimeHeadsv3](https://huggingface.co/datasets/nyuuzyou/AnimeHeadsv3)**: Augmented COCO-format anime heads dataset.

---

## Documentation

- **[`docs/architecture.md`](docs/architecture.md)** — Machine learning design, Label Studio class remapping, and scale parity rationale.
- **[`docs/runbook.md`](docs/runbook.md)** — The operational runbook: data pipeline, pre-slicing, Cloudflare R2 storage, and training flows.

---

## Credits

- [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics)
- [SAHI: Slicing Aided Hyper Inference](https://github.com/obss/sahi)
- [Fuyucch1/yolov8_animeface](https://github.com/Fuyucch1/yolov8_animeface)