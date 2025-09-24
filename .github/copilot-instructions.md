# AI Agent Instructions - Manga Character Finder

## Project Overview
This is a computer vision pipeline for finding and retrieving similar manga characters. The project processes manga pages through a 4-stage sequential pipeline implemented in a Jupyter notebook (`mangaCharRetrieval.ipynb`).

## Core Architecture & Data Flow
The pipeline follows this strict sequence (cells must be run in order):

1. **Panel Extraction** → `data/panels/` 
   - Hybrid approach: OpenCV contours + Hough line detection for gutter splitting
   - Handles complex manga layouts with irregular panel shapes
   
2. **Character Detection** → `data/crops/`
   - Dual approach: Heuristic blob detection + YOLOv8 anime face detection
   - YOLOv8 model: `yolov8x6_animeface.pt` (not in git, downloaded separately)
   
3. **Embedding Generation** → `data/embeddings.npy` + `data/crop_paths.json`
   - TensorFlow MobileNetV2 backbone with custom L2-normalized 256D output layer
   - Batch processing at IMG_SIZE=128px
   
4. **Similarity Search** → Visual results via matplotlib
   - scikit-learn NearestNeighbors with cosine distance
   - Returns top-N similar character crops

## Environment & Dependencies
- **Package Manager**: Pixi (conda-forge based)
- **Platform**: macOS ARM64 (osx-arm64)
- **Key Dependencies**: TensorFlow, OpenCV, PyTorch/Ultralytics, scikit-learn
- **Setup**: `pixi install` (no manual pip/conda needed)

## Critical File Structure
```
data/
├── pages/          # Input: Raw manga page images (JPG/PNG)
├── panels/         # Stage 1: Extracted panels
├── crops/          # Stage 2: Character face/head candidates
├── embeddings.npy  # Stage 3: Feature vectors (N x 256)
└── crop_paths.json # Stage 3: Mapping indices to file paths
```

## Key Code Patterns

### Naming Conventions
- Panel files: `{page_stem}_panel_{i}.jpg` or `{page_stem}_panel_{i}_{j}.jpg` (if split)
- Crop files: `{panel_stem}_crop_{component_id}.jpg` (heuristic) or `{panel_stem}_face_{i}.jpg` (YOLO)
- Directories created automatically via `os.makedirs(exist_ok=True)`

### Configuration Variables
Always update these in the first cell:
```python
PAGES_DIR = 'data/pages'  # Point to your manga images
PANELS_DIR = 'data/panels'
CROPS_DIR = 'data/crops'
```

### Critical Thresholds
- Panel extraction: `min_area=20000` (filters small noise)
- Heuristic detection: `area < 800` (minimum blob size), `0.35 < aspect < 1.8` (head-like ratios)
- YOLO confidence: `conf=0.3`, `imgsz=512`

## Development Workflow

### Running the Pipeline
1. Place manga pages in `data/pages/`
2. Execute notebook cells sequentially (never skip or reorder)
3. Check output counts after each stage
4. For search queries, update `seed_path` in final cell

### Debugging Common Issues
- Empty crops: Check YOLO model file exists and confidence threshold
- No panels extracted: Verify image paths and `min_area` threshold
- Embedding errors: Ensure crops directory has images before running embedding cell
- Search failures: Verify `embeddings.npy` and `crop_paths.json` exist and match

### Model Dependencies
- YOLOv8 model must be downloaded separately (not in repository)
- TensorFlow MobileNetV2 uses ImageNet pretrained weights (auto-downloaded)
- All models expect RGB input (PIL/OpenCV conversions handled automatically)

## Integration Points
- **Input**: Expects JPG/PNG manga pages in `data/pages/`
- **Output**: Visual similarity grid via matplotlib (5x6 subplot layout)
- **Persistence**: Embeddings cached as `.npy` + JSON index for reuse
- **External Models**: YOLOv8 (Ultralytics) + MobileNetV2 (TensorFlow/Keras)

## Project-Specific Notes
- Project originally named "izutsumi-finder" (character from Dungeon Meshi)
- Sequential processing design - each stage depends on previous outputs
- Hybrid detection approach compensates for different manga art styles
- Cosine similarity preferred over Euclidean for normalized embeddings