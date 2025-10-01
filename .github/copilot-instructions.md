# Catnip - AI Assistant Instructions

This is a manga character recognition system that extracts specific characters from manga pages using computer vision and deep learning. The workflow is designed around Jupyter notebooks and uses Pixi for environment management.

## Core Architecture & Data Flow

**Processing Pipeline:**
1. **Panel Extraction** (`panelExtraction.py`) - Extract panels from manga pages using contours + Hough line detection
2. **Head/Face Extraction** (`headExtraction.py`) - Extract character faces using YOLOv8 anime face detection or heuristic blob detection
3. **Feature Embedding** (`recognitionModel.py`) - Generate embeddings using MobileNetV2 + L2 normalization
4. **Character Search** - Find character matches using cosine similarity threshold search

**Data Structure:**
```
data/
├── pages/           # Raw manga page images (input)
├── panels/          # Extracted panels from pages
├── crops/           # Character face/head crops from panels
├── curated_pages/   # Manual curated dataset
├── izutsumi/        # Seed images for target character
├── embeddings.npy   # Precomputed feature embeddings
└── crop_paths.json  # Maps embedding indices to crop file paths
```

## Environment & Package Management

- **Use Pixi** (not pip/conda directly): `pixi install`, `pixi shell`, `pixi run`
- **Key dependencies**: TensorFlow (MobileNetV2), Ultralytics (YOLOv8), OpenCV, scikit-learn
- **Special dependency**: `adenzu-manga-panel-extractor-src` installed from git in `pixi.toml`
- **Configuration**: Environment paths defined in `paths.env`, loaded via Pydantic Settings in `src/config.py`

## Critical Project-Specific Patterns

**Configuration Management:**
- Use `src.config.settings` object for all paths - never hardcode data paths
- Settings auto-loads from `paths.env` file using Pydantic BaseSettings
- Directory structure is enforced by settings: `pages_dir`, `panels_dir`, `crops_dir`, etc.

**Image Processing Conventions:**
- **IMG_SIZE = 128** for embeddings (fixed throughout pipeline)
- **Similarity threshold 0.7-0.8** typical range (0.8 = very similar, 0.6 = somewhat similar)
- **Batch processing**: Always process images in batches with progress bars (tqdm)
- **File naming**: `{source}_{type}_{id}.jpg` pattern (e.g., `page1_panel_0_face_2.jpg`)

**Memory Management:**
- Embeddings computed in streaming fashion to avoid memory overload
- Use batch prediction with `batch_size=64` for TensorFlow models
- Large datasets processed with generators, not loaded entirely into memory

## Development Workflow

**Primary Interface**: `catnip.ipynb` notebook with 5 main cells:
1. Setup directories and configuration
2. Extract panels using adenzu panel extractor (slow but accurate)
3. Extract character faces using YOLOv8 anime face detection
4. Build embedding model and compute features for all crops
5. Search for character matches using seed images and similarity threshold

**Alternative Panel Extraction**: `hybrid_extract_panels()` in `panelExtraction.py` - faster but less accurate than adenzu extractor

**Testing New Characters**: 
- Add seed images to `data/izutsumi/` directory
- Adjust `similarity_threshold` in notebook (higher = stricter matching)
- Results displayed in matplotlib grid with similarity scores

## Model Files & External Dependencies

**YOLOv8 Model**: `src/models/yolov8x6_animeface.pt` (Fuyucch1/yolov8_animeface)
- Pre-trained on anime/manga faces
- Input size 512px, confidence 0.3, IoU 0.5 for inference

**MobileNetV2 Embedding**: 
- ImageNet pretrained backbone → GlobalAveragePooling2D → Dense(1024) → L2 normalization
- Generates unit-length embeddings for cosine similarity search

## Common Operations

**Full Pipeline Run:**
```python
# Run all cells in catnip.ipynb sequentially
# OR run individual modules:
from src.headExtraction import anime_extraction_recursive
from src.recognitionModel import build_model, char_nearest_neighbor
```

**Quick Character Search:**
```python
# After embeddings are computed once:
char_nearest_neighbor(settings.embed_path, settings.crop_path, 128, embed_model, seed_paths, similarity_threshold=0.8)
```

**Directory Setup:**
```python
from src.config import settings
os.makedirs(settings.panels_dir, exist_ok=True)  # Always use settings paths
```

## Key Integration Points

- **Adenzu Panel Extractor**: `modules/coreMPE/` - external git submodule for accurate panel extraction
- **YOLOv8**: Ultralytics integration for anime face detection with custom model weights
- **TensorFlow/Keras**: MobileNetV2 feature extraction with custom head architecture
- **Environment Files**: `paths.env` → `src.config.Settings` → all modules (never hardcode paths)

When modifying this codebase, maintain the streaming processing approach for large datasets and always use the settings object for path management.