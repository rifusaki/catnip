from pathlib import Path
from omegaconf import OmegaConf
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


class Paths(BaseModel):
    data: Path

    # Source Data
    manga_dir: Path
    annotations_dir: Path
    ls_exports_dir: Path

    # Stage 1: YOLO Detection
    stage1_dir: Path
    stage1_images_dir: Path
    stage1_labels_dir: Path
    crops_dir: Path

    # Stage 2: Re-ID
    stage2_dir: Path
    izutsumi_dir: Path
    not_izutsumi_dir: Path

    # External Datasets
    animeheads_dir: Path
    ah_v1_dir: Path
    ah_v2_dir: Path
    ah_af_dir: Path
    ah_coco_dir: Path
    manga109_dir: Path
    izutsumi_manga_dir: Path
    izutsumi_annotations_dir: Path

    # Unified Training
    unified_dir: Path
    unified_images_dir: Path
    unified_labels_dir: Path

    # Pre-sliced (SAHI-parity) Training
    stage1_sliced_dir: Path

    # Models & Outputs
    model_dir: Path
    runs_dir: Path
    output_dir: Path


class Stage1Labels(BaseModel):
    body: int
    head: int
    face: int


class Stage2Labels(BaseModel):
    izutsumi: int
    not_izutsumi: int


class Labels(BaseModel):
    stage1: Stage1Labels
    stage2: Stage2Labels


class SahiParams(BaseModel):
    slice_height: int
    slice_width: int
    overlap_ratio: float
    confidence_threshold: float
    min_area_ratio: float


class MetricLearningParams(BaseModel):
    img_size: int
    batch_size: int
    margin: float
    learning_rate: float
    epochs: int
    faiss_index_path: str
    embeddings_model_path: str


class Params(BaseModel):
    device: str
    sahi: SahiParams
    metric_learning: MetricLearningParams


class Stage1TrainingParams(BaseModel):
    model: str
    imgsz: int
    epochs: int
    patience: int
    batch: int
    workers: int
    device: str
    freeze: int
    mosaic: float
    lr0: float
    cos_lr: bool


class Stage2TrainingParams(BaseModel):
    model: str
    batch_size: int
    epochs: int
    learning_rate: float
    patience: int
    device: str
    margin: float
    p: int
    k: int


class Training(BaseModel):
    stage1: Stage1TrainingParams
    stage2: Stage2TrainingParams


class Settings(BaseModel):
    paths: Paths
    params: Params
    labels: Labels
    training: Training


def load_settings(path: str | Path = "config/pipeline.yaml") -> Settings:
    root = Path.cwd()
    config_path = root / path

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    cfg = OmegaConf.load(config_path)

    # Resolve all ${...} interpolations
    resolved = OmegaConf.to_container(cfg, resolve=True)
    return Settings(**resolved)


_settings: Settings | None = None


def get_settings(path: str | Path = "config/pipeline.yaml") -> Settings:
    """Return the cached Settings singleton, loading on first call."""
    global _settings
    if _settings is None:
        _settings = load_settings(path)
    return _settings


# Legacy module-level alias — resolved lazily on first attribute access via
# a thin proxy so that existing callers (`from src.config import settings`)
# continue to work without triggering OmegaConf at import time.
class _LazySettings:
    """Proxy that forwards attribute access to the real Settings on demand."""
    def __getattr__(self, name: str):
        return getattr(get_settings(), name)

    def __repr__(self) -> str:
        return repr(get_settings())


settings: Settings = _LazySettings()  # type: ignore[assignment]


def setup_dirs():
    """Create necessary directories and scan for existing images."""
    for path in vars(settings.paths).values():
        if isinstance(path, Path) and path.suffix == "":
            path.mkdir(parents=True, exist_ok=True)

    izutsumi = []
    not_izutsumi = []

    if settings.paths.izutsumi_dir and settings.paths.izutsumi_dir.exists():
        izutsumi = [str(p) for p in sorted(
            list(settings.paths.izutsumi_dir.glob("*.jpg"))
            + list(settings.paths.izutsumi_dir.glob("*.jpeg"))
            + list(settings.paths.izutsumi_dir.glob("*.png"))
        )]

    if settings.paths.not_izutsumi_dir and settings.paths.not_izutsumi_dir.exists():
        not_izutsumi = [str(p) for p in sorted(
            list(settings.paths.not_izutsumi_dir.glob("*.jpg"))
            + list(settings.paths.not_izutsumi_dir.glob("*.jpeg"))
            + list(settings.paths.not_izutsumi_dir.glob("*.png"))
        )]

    return izutsumi, not_izutsumi
