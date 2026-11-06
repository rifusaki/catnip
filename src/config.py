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
    stage1_sliced_subset_dir: Path

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
    # Post-slicing fusion: WBF fuses overlapping slice predictions by
    # confidence-weighted box averaging (NMM keeps the largest box).
    postprocess_type: str = "WBF"
    postprocess_match_metric: str = "IOS"
    postprocess_match_threshold: float = 0.40
    # Dynamic panel/page gating (see SAHIExtractor.extract_from_image):
    # images with max(H, W) <= panel_max_dim skip SAHI and run a single
    # letterboxed 640px inference pass.
    panel_max_dim: int = 800
    # Full pages are normalized so the longer side equals this before
    # slicing, then predictions are projected back to original pixel space.
    canonical_page_max_dim: int = 1800
    # Hierarchical anatomy filter: low-confidence face boxes are kept only
    # when contained (IoS >= 0.75) inside a detected head or body box.
    hierarchical_verify: bool = True
    # Per-class calibrated confidence thresholds applied after fusion.
    class_conf_thresholds: dict[str, float] = {
        "body": 0.52,
        "head": 0.52,
        "face": 0.56,
    }


class ReidParams(BaseModel):
    """Stage 2 character embedding settings (multi-prototype clustering)."""

    # Backbone registry key resolved by src/reid/model.py:
    # "convnext_small" (torchvision, default), "dinov2_vits14" (torch.hub),
    # "ccip" (deepghs, optional dependency).
    backbone: str = "convnext_small"
    embedding_dim: int = 256
    # Generalized Mean pooling exponent (3.0 = power mean, learnable).
    gem_p: float = 3.0
    img_size: int = 224
    # Confidence-weighted part fusion weights ({face, head, body}).
    # Renormalized over present parts at inference time.
    part_weights: dict[str, float] = {"face": 0.35, "head": 0.50, "body": 0.15}
    # FAISS index artifact (IndexFlatIP over L2-normalized embeddings).
    index_path: str = "reid_index.faiss"


class ClusteringParams(BaseModel):
    """Semi-supervised multi-prototype clustering and decision thresholds."""

    # k-NN graph edges per instance for HDBSCAN density estimation.
    k_neighbors: int = 25
    # HDBSCAN minimum sub-cluster membership.
    min_cluster_size: int = 5
    # Tier 1 auto-tag threshold (target precision >= 98%).
    tau_core: float = 0.82
    # Tier 2 boundary band lower edge: [tau_boundary, tau_core) instances
    # are grouped into review sub-clusters instead of auto-tagged.
    tau_boundary: float = 0.68
    # Same-Page Prior Boost added to partial/back instances in adjacent
    # panels of a page that already contains a core Izutsumi detection.
    page_prior_boost: float = 0.06
    # Two spatially distinct instances with IoU below this are considered
    # distinct characters for Same-Panel Cannot-Link.
    panel_instance_iou: float = 0.15


class Params(BaseModel):
    device: str
    sahi: SahiParams
    reid: ReidParams
    clustering: ClusteringParams


class SubsetSourceParams(BaseModel):
    """Per-source sampling knobs for the stratified stage1 subset."""
    fraction: float
    max_per_image: int  # 0 = unlimited


class Stage1SubsetParams(BaseModel):
    seed: int
    train: dict[str, SubsetSourceParams]
    val: dict[str, SubsetSourceParams]


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
    optimizer: str
    use_subset: bool
    deterministic: bool
    subset: Stage1SubsetParams
    # Optional run/artifact naming: when empty, stage1.py derives the
    # historical defaults ("stage1"/"stage1_m3" run dirs and target names).
    # Non-empty names keep v2 runs (yolo26m) from clobbering v1 artifacts.
    run_name: str = ""
    target_model_name: str = ""
    close_mosaic: int = 10
    weight_decay: float = 0.0005
    cls_pw: float = 0.0
    box: float = 7.5
    dfl: float = 1.5
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    # Color-to-line-art bridge: probability of RandomGrayscale on
    # training batches (0 disables; Stage 1 v2 uses 0.25).
    random_grayscale: float = 0.0
    mixup: float = 0.0
    plots: bool = True
    save_period: int = -1


class Stage2TrainingParams(BaseModel):
    """SupCon / Sub-Center ArcFace training (semi-supervised clustering)."""

    model: str
    backbone: str = "convnext_small"
    embedding_dim: int = 256
    batch_size: int
    epochs: int
    learning_rate: float
    patience: int
    device: str
    # Supervised Contrastive Loss temperature (tau).
    temperature: float = 0.07
    # Sub-Center ArcFace sub-centers per class (Izutsumi visual modes).
    num_subcenters: int = 5
    image_size: int = 224


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
