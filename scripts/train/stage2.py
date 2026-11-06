#!/usr/bin/env python3
"""
Stage 2 Re-ID Fine-Tuning — SupCon + Sub-Center ArcFace (plan §5.2).

Semi-supervised multi-prototype training: instead of collapsing every
"not_izutsumi" crop into one binary negative, crops are embedded with a
pretrained backbone, HDBSCAN mines pseudo-cluster IDs for the recurring
non-Izutsumi characters (noise points become unique repulsion-only classes),
and Izutsumi (multi-modal) keeps a single label 0 whose visual modes are
absorbed by K ArcFace sub-centers.

Usage::

    python scripts/train/stage2.py                        # train (crops required)
    python scripts/train/stage2.py --prepare-only        # data prep only, exit 0
    python scripts/train/stage2.py --override training.stage2.epochs=200
    python scripts/train/stage2.py --verbose

The default flow downloads pretrained backbone weights (torch.hub on Colab);
tests always prepare crops and call the exported helpers with prebuilt
embedders instead.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

# Allow running the script directly from scripts/train/ without an installed package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import Settings
from src.reid.loss import SubCenterArcFace, SupConLoss
from src.reid.model import CharacterEmbedder, preprocess_rgb, save_embedder
from src.training.preparation import prepare_triplet_dataset

logger = logging.getLogger(__name__)

# Reproducibility: matches prepare_triplet_dataset's split seed.
TRAIN_SEED = 42

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
IZUTSUMI_LABEL = 0
CLASS_MAP_FILENAME = "class_map.json"


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------

def _configure_logging(verbose: bool) -> None:
    """Set up logging: INFO by default, DEBUG when --verbose."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(message)s",
    )


def _load_and_override_config(
    config_path: str, overrides: list[str] | None
) -> Settings:
    """Parse config YAML, apply CLI overrides, return typed Settings.

    Law 4 (Fail Fast): exits immediately if the config file is missing.
    Law 2 (Parse at boundary): OmegaConf → Pydantic so internals trust the shape.

    NOTE (deliberate difference from src.cli._apply_overrides): override
    values are passed to OmegaConf as RAW STRINGS and rely on pydantic's
    lax coercion for the final type cast. Booleans and non-numeric values
    keep working because pydantic coerces "true"/"false"/numbers; numeric
    strings land as str only when the schema field is str. This keeps the
    dodgy-type-parsing of the CLI out of the training path deliberately —
    the config schema is trusted here, mirroring _load_and_override_config
    in the original placeholder.
    """
    root = Path.cwd()
    full_path = root / config_path

    # --- Guard: config file must exist ---
    if not full_path.exists():
        logger.error("Configuration file not found: %s", full_path)
        sys.exit(1)

    # --- Load raw OmegaConf ---
    cfg = OmegaConf.load(full_path)

    # --- Apply CLI overrides (dot-path KEY=VALUE) ---
    if overrides:
        for override in overrides:
            if "=" not in override:
                logger.warning(
                    "Skipping malformed override (missing '='): %s", override
                )
                continue
            key, value = override.split("=", 1)
            try:
                OmegaConf.update(cfg, key, value, merge=True)
                logger.debug("Override applied: %s = %s", key, value)
            except Exception as exc:
                logger.error(
                    "Failed to apply override '%s': %s", override, exc
                )
                sys.exit(1)

    # --- Resolve interpolations and convert to Pydantic ---
    resolved = OmegaConf.to_container(cfg, resolve=True)
    return Settings(**resolved)


def _log_training_config(settings: Settings) -> None:
    """Print the Stage 2 SupCon / Sub-Center ArcFace training profile."""
    s2 = settings.training.stage2
    reid = settings.params.reid

    logger.info("=== Stage 2 SupCon ReID Training ===")
    logger.info("  model:            %s", s2.model)
    logger.info("  backbone:         %s (reid params: %s)", s2.backbone, reid.backbone)
    logger.info("  embedding_dim:    %d (reid params: %d)", s2.embedding_dim, reid.embedding_dim)
    logger.info("  batch_size:       %d", s2.batch_size)
    logger.info("  epochs:           %d", s2.epochs)
    logger.info("  learning_rate:    %.6f", s2.learning_rate)
    logger.info("  device:           %s", s2.device)
    logger.info("  temperature:      %.4f (SupConLoss tau)", s2.temperature)
    logger.info("  sub-centers:      %d (Sub-Center ArcFace K)", s2.num_subcenters)
    logger.info("  image_size:       %d (model canonical size: %d)",
                s2.image_size, reid.img_size)
    logger.info("  patience:         %d", s2.patience)
    logger.info("  crops dir:        %s", settings.paths.stage2_dir)
    logger.info("  index artifact:   %s", reid.index_path)
    logger.info("")


# ---------------------------------------------------------------------------
# Data prep path (--prepare-only)
# ---------------------------------------------------------------------------

def _prepare_dataset(settings: Settings) -> dict:
    """Run ``prepare_triplet_dataset()`` and return the count dictionary.

    Law 1 (Early Exit): warns and returns empty counts when crops_dir is
    missing instead of crashing.
    """
    crops_dir = settings.paths.crops_dir

    # --- Guard: crops_dir may not exist yet ---
    if not crops_dir.exists():
        logger.warning(
            "Crops directory does not exist: %s — dataset will have 0 images.",
            crops_dir,
        )
        return {
            "train": {"izutsumi": 0, "not_izutsumi": 0},
            "val": {"izutsumi": 0, "not_izutsumi": 0},
        }

    logger.info("Preparing triplet dataset from crops...")
    result = prepare_triplet_dataset(
        source_dirs=[crops_dir],
    )

    return result


def _log_dataset_counts(counts: dict) -> None:
    """Print per-split image counts discovered by dataset preparation."""
    train = counts.get("train", {})
    val = counts.get("val", {})

    logger.info("=== Triplet Dataset Counts ===")
    logger.info(
        "  Train — izutsumi: %d, not_izutsumi: %d",
        train.get("izutsumi", 0),
        train.get("not_izutsumi", 0),
    )
    logger.info(
        "  Val   — izutsumi: %d, not_izutsumi: %d",
        val.get("izutsumi", 0),
        val.get("not_izutsumi", 0),
    )

    total = sum(train.values()) + sum(val.values())
    logger.info("  Total: %d images", total)
    logger.info("")


# ---------------------------------------------------------------------------
# Crop discovery & embedding
# ---------------------------------------------------------------------------

def _discover_split(stage2_dir: Path, split: str) -> dict[str, list[Path]]:
    """List izutsumi / not_izutsumi crop paths for one split under stage2_dir."""
    crops: dict[str, list[Path]] = {}
    for class_name in ("izutsumi", "not_izutsumi"):
        class_dir = stage2_dir / split / class_name
        if not class_dir.exists():
            logger.warning("Stage 2 %s/%s directory missing: %s", split, class_name, class_dir)
            crops[class_name] = []
            continue
        crops[class_name] = sorted(
            path
            for path in class_dir.iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file()
        )
    return crops


def _load_rgb(path: Path) -> np.ndarray:
    """Read an image as uint8 RGB; fail loud on unreadable files."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def embed_dataset(embedder: CharacterEmbedder, image_paths: list) -> np.ndarray:
    """Embed a list of crop paths with ``embed_single`` -> [N, embedding_dim].

    Rows are unit-length (embed_single output); an empty path list yields a
    (0, embedding_dim) array.
    """
    image_paths = list(image_paths)
    if not image_paths:
        return np.zeros((0, embedder.embedding_dim), dtype=np.float32)
    return np.stack([embedder.embed_single(_load_rgb(path)) for path in image_paths])


# ---------------------------------------------------------------------------
# Pseudo-label mining (HDBSCAN over the unit sphere)
# ---------------------------------------------------------------------------

def angular_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Pairwise angular distance arccos(clip(<a, b> in [-1, 1]))."""
    cosines = np.clip(embeddings @ embeddings.T, -1.0, 1.0)
    return np.arccos(cosines)


def mine_pseudo_labels(embeddings: np.ndarray, min_cluster_size: int = 5) -> list[int]:
    """HDBSCAN pseudo-cluster IDs for not_izutsumi embeddings (labels never 0).

    Density clusters (recurring characters like Laios/Marcille) receive shared
    labels 1..C in sorted cluster order; HDBSCAN noise points become UNIQUE
    singletons (C+1, C+2, ...) so background NPCs act as pure repulsion
    negatives instead of an artificial mega-cluster.  Label 0 stays reserved
    for izutsumi.

    min_cluster_size=5 matches settings.params.clustering.min_cluster_size: the
    smallest membership a character sub-cluster must have to be treated as a
    recurring character rather than one-off background noise.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be [N, D], got {tuple(embeddings.shape)}")
    n_samples = embeddings.shape[0]

    # Too few crops to form any cluster: everyone is its own repulsion class.
    if n_samples < max(min_cluster_size, 2):
        return list(range(1, n_samples + 1))

    normalized = embeddings / np.clip(
        np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12, None
    )

    # The dense precomputed distance matrix is O(N^2): 50k crops would be
    # ~20 GB. Refuse loudly instead of OOMing a multi-hour Colab run; the
    # budget matches src.reid.clustering.MAX_SUBCLUSTER_INSTANCES.
    from src.reid.clustering import MAX_SUBCLUSTER_INSTANCES

    if n_samples > MAX_SUBCLUSTER_INSTANCES:
        raise ValueError(
            f"{n_samples} not_izutsumi crops exceed the dense precomputed "
            f"HDBSCAN budget ({MAX_SUBCLUSTER_INSTANCES}x{MAX_SUBCLUSTER_INSTANCES} "
            "angular distance matrix, ~1.6 GB at float32). Subsample repetitive "
            "crops or mine in embedding-space batches before training."
        )

    distances = angular_distance_matrix(normalized)

    from sklearn.cluster import HDBSCAN

    fit = HDBSCAN(min_cluster_size=min_cluster_size, metric="precomputed").fit(distances)
    raw_labels = np.asarray(fit.labels_)

    pseudo_labels = np.zeros(n_samples, dtype=np.int64)
    cluster_ids = np.unique(raw_labels)
    cluster_ids = cluster_ids[cluster_ids != -1]

    next_label = 1
    for cluster_id in cluster_ids:
        pseudo_labels[raw_labels == cluster_id] = next_label
        next_label += 1
    # Noise -> one unique label per point, in index order (pure repulsion).
    for index in np.flatnonzero(raw_labels == -1):
        pseudo_labels[index] = next_label
        next_label += 1

    return pseudo_labels.astype(int).tolist()


# ---------------------------------------------------------------------------
# Validation & training loop
# ---------------------------------------------------------------------------

def eval_separability(embeddings: np.ndarray, is_izutsumi: list[bool]) -> float:
    """Mean izutsumi-to-izutsumi cosine minus mean izutsumi-to-other cosine.

    Require: >= 2 izutsumi embeddings (for an intra-class mean) and >= 1
    other embedding (for the inter-class mean); anything else is undefined
    and fails loud.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    flags = np.asarray(is_izutsumi, dtype=bool)
    if embeddings.ndim != 2 or flags.shape[0] != embeddings.shape[0]:
        raise ValueError(
            f"embeddings {tuple(embeddings.shape)} and is_izutsumi {flags.shape[0]} disagree"
        )
    izu = embeddings[flags]
    others = embeddings[~flags]
    if len(izu) < 2 or len(others) < 1:
        raise ValueError(
            f"separability needs >= 2 izutsumi and >= 1 other embeddings, got "
            f"{len(izu)} and {len(others)}"
        )

    intra_pairs = izu @ izu.T
    triu_indices = np.triu_indices(len(izu), k=1)
    intra_mean = intra_pairs[triu_indices].mean()
    inter_mean = (izu @ others.T).mean()
    return float(intra_mean - inter_mean)


def _preload_images(image_paths: list[Path]) -> list[np.ndarray]:
    """Decode every training crop once (native size) -> uint8 RGB arrays.

    Native-resolution caching keeps training preprocessing IDENTICAL to
    deployment (preprocess_rgb resize) at the cost of one image per crop in
    RAM — sized for crop datasets (tens of thousands of crops).
    """
    images: list[np.ndarray] = []
    failures: list[Path] = []
    for path in image_paths:
        try:
            images.append(_load_rgb(path))
        except ValueError:
            failures.append(path)

    if failures:
        raise ValueError(
            f"{len(failures)} unreadable crop images; first failures: "
            f"{[str(path) for path in failures[:5]]}"
        )
    return images


def train(
    model: CharacterEmbedder,
    train_paths: list[Path],
    train_labels: list[int],
    val_paths: list[Path] | None,
    val_labels: list[int] | None,
    *,
    num_classes: int,
    device: torch.device,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    patience: int,
    temperature: float,
    num_subcenters: int,
    model_path: Path,
) -> dict:
    """Fine-tune ``model`` with SupCon + Sub-Center ArcFace (joint, 1:1).

    Saves the best checkpoint (by val separability) to ``model_path`` via
    ``save_embedder`` on completion.  With no usable validation split the
    final model is saved after the last epoch.  Returns the metric history,
    including ``best_val_separability``.
    """
    # --- Law 1: guard clauses before any state is built
    if len(train_paths) != len(train_labels) or not train_paths:
        raise ValueError(
            f"train paths ({len(train_paths)}) and labels ({len(train_labels)}) "
            "must be non-empty and aligned"
        )
    if val_paths is not None and len(val_paths) != len(val_labels):
        raise ValueError("val paths and val labels must be aligned")
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2 (izutsumi + pseudo), got {num_classes}")
    if batch_size < 1 or epochs < 1:
        raise ValueError(f"batch_size and epochs must be >= 1, got {batch_size} / {epochs}")

    use_validation = bool(val_paths)
    if use_validation:
        # Law 1: separability requirements are checked up-front, not per epoch.
        val_is_izutsumi = [label == IZUTSUMI_LABEL for label in val_labels]
        izu_count = sum(val_is_izutsumi)
        if izu_count < 2 or (len(val_is_izutsumi) - izu_count) < 1:
            raise ValueError(
                f"validation split unusable: separability needs >= 2 "
                f"izutsumi and >= 1 other crop, got {izu_count} and "
                f"{len(val_is_izutsumi) - izu_count}"
            )

    crop_cache = _preload_images(train_paths)
    num_crops = len(crop_cache)
    img_size = model.img_size

    model = model.to(device)
    model.train()

    supcon = SupConLoss(temperature=temperature)
    arcface = SubCenterArcFace(
        in_features=model.embedding_dim,
        out_features=num_classes,
        num_subcenters=num_subcenters,
    ).to(device)

    optimizer = torch.optim.AdamW(
        itertools.chain(model.parameters(), arcface.parameters()), lr=learning_rate
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    shuffle_generator = torch.Generator()

    history: dict[str, list[float]] = {
        "train_loss": [],
        "loss_supcon": [],
        "loss_arcface": [],
        "val_separability": [],
    }
    best_separability: float | None = None
    best_state: dict | None = None
    stalled_epochs = 0

    for epoch in range(1, epochs + 1):
        shuffle_generator.manual_seed(TRAIN_SEED + epoch)
        order = torch.randperm(num_crops, generator=shuffle_generator).tolist()

        epoch_supcon = 0.0
        epoch_arcface = 0.0
        batches = 0
        for start in range(0, num_crops, batch_size):
            batch_index = order[start : start + batch_size]
            image_batch = torch.stack(
                [preprocess_rgb(crop_cache[i], img_size) for i in batch_index]
            ).to(device)
            label_batch = torch.tensor(
                [train_labels[i] for i in batch_index], dtype=torch.long, device=device
            )

            embeddings = model(image_batch)
            batch_supcon = supcon(embeddings, label_batch)
            logits, batch_arcface = arcface(embeddings, label_batch)
            loss = batch_supcon + batch_arcface

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_supcon += float(batch_supcon.detach())
            epoch_arcface += float(batch_arcface.detach())
            batches += 1

        scheduler.step()
        epoch_loss = epoch_supcon + epoch_arcface
        history["train_loss"].append(epoch_loss / batches)
        history["loss_supcon"].append(epoch_supcon / batches)
        history["loss_arcface"].append(epoch_arcface / batches)

        if not use_validation:
            logger.info(
                "epoch %3d/%d — train loss %.4f (supcon %.4f + arcface %.4f)",
                epoch, epochs, epoch_loss / batches,
                epoch_supcon / batches, epoch_arcface / batches,
            )
            continue

        model.eval()
        val_embeddings = embed_dataset(model, val_paths)
        separability = eval_separability(
            val_embeddings, [label == IZUTSUMI_LABEL for label in val_labels]
        )
        model.train()
        history["val_separability"].append(separability)
        logger.info(
            "epoch %3d/%d — train loss %.4f (supcon %.4f + arcface %.4f), val separability %.4f",
            epoch, epochs, epoch_loss / batches,
            epoch_supcon / batches, epoch_arcface / batches, separability,
        )
        if best_separability is None or separability > best_separability:
            best_separability = separability
            best_state = {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            }
            stalled_epochs = 0
        else:
            stalled_epochs += 1
            if stalled_epochs >= patience:
                logger.info(
                    "Early stopping: %d epochs without separability improvement "
                    "(patience %d)", stalled_epochs, patience
                )
                break

    if use_validation and history["train_loss"]:
        logger.info("Best validation separability: %.4f", best_separability)

    # --- Save best model once (best state, or final state without validation)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    save_embedder(model, model_path)
    logger.info("Saved Stage 2 embedder to %s", model_path)

    return {**history, "best_val_separability": best_separability}


# ---------------------------------------------------------------------------
# Device resolution & embedder construction
# ---------------------------------------------------------------------------

def _resolve_device(name: str) -> torch.device:
    """Configured device name -> available torch.device (cpu fallback + warn)."""
    wanted = name.lower()
    if wanted == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if wanted == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if wanted != "cpu":
        logger.warning("Device '%s' unavailable — falling back to CPU", name)
    return torch.device("cpu")


def build_embedder(settings: Settings) -> CharacterEmbedder:
    """Pretrained mining embedder from reid params (network: Colab path only).

    Training proceeds with ``settings.training.stage2`` values; disagreements
    between the stage2 and reid config blocks are surfaced as warnings if both
    are explicitly configured, since the reid block wins contractually.
    """
    reid = settings.params.reid
    s2 = settings.training.stage2
    if s2.backbone != reid.backbone:
        logger.warning(
            "training.stage2.backbone '%s' differs from params.reid.backbone '%s' "
            "— using the reid block", s2.backbone, reid.backbone
        )

    return CharacterEmbedder(
        backbone=reid.backbone,
        embedding_dim=reid.embedding_dim,
        gem_p=reid.gem_p,
        pretrained=True,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the stage2 CLI. ``--override`` is repeatable, matching the
    repo convention (one KEY=VALUE per flag, flags accumulate)."""
    parser = argparse.ArgumentParser(
        description="Stage 2 SupCon / Sub-Center ArcFace ReID fine-tuning.",
    )
    parser.add_argument(
        "--config",
        default="config/pipeline.yaml",
        help="Path to pipeline configuration YAML (default: %(default)s).",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Override config values using dot-path keys, repeatable "
             "(e.g. --override training.stage2.epochs=200)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only run triplet dataset preparation (legacy data-prep path, exit 0).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_cli_args(argv)

    # --- Setup logging ---
    _configure_logging(args.verbose)

    # --- Load and validate configuration (Law 4 — Fail Fast) ---
    settings = _load_and_override_config(args.config, args.override)

    # --- Log training hyperparameters ---
    _log_training_config(settings)

    if args.prepare_only:
        counts = _prepare_dataset(settings)
        _log_dataset_counts(counts)
        logger.info("Dataset preparation complete (--prepare-only).")
        sys.exit(0)

    _run_training(settings)


def _run_training(settings: Settings) -> None:
    """The real Stage 2 flow: discover -> embed -> mine -> train -> save."""
    # --- Guard: training crops must exist BEFORE any network call (Law 4 —
    # tests and offline runs must fail here, not at weight download)
    train_split_dir = settings.paths.stage2_dir / "train"
    if not train_split_dir.exists():
        logger.error(
            "Stage 2 training crops not found — %s does not exist. Run "
            "`python scripts/train/stage2.py --prepare-only` (or "
            "`pixi run train-stage2 --prepare-only`) to build the crop "
            "splits from Stage 1 detections first.",
            train_split_dir,
        )
        sys.exit(1)

    train_crops = _discover_split(settings.paths.stage2_dir, "train")
    val_crops = _discover_split(settings.paths.stage2_dir, "val")

    izutsumi_paths = train_crops["izutsumi"]
    other_paths = train_crops["not_izutsumi"]
    if not izutsumi_paths or not other_paths:
        logger.error(
            "Stage 2 training requires both classes under %s — found "
            "izutsumi=%d, not_izutsumi=%d. Re-run data preparation.",
            settings.paths.stage2_dir / "train",
            len(izutsumi_paths), len(other_paths),
        )
        sys.exit(1)

    torch.manual_seed(TRAIN_SEED)
    device = _resolve_device(settings.training.stage2.device)

    # --- Pretrained mining embedder (downloads weights: Colab-only path) ---
    embedder = build_embedder(settings).to(device)
    embedder.eval()
    logger.info(
        "Mining pseudo-labels: embedding %d not_izutsumi crops (%s, %s)",
        len(other_paths), embedder.backbone_name, device,
    )

    # --- Pseudo-cluster mining over not_izutsumi embeddings (singletons stay
    # separate repulsion classes; izutsumi keeps reserved label 0)
    other_embeddings = embed_dataset(embedder, other_paths)
    other_labels = mine_pseudo_labels(
        other_embeddings,
        min_cluster_size=settings.params.clustering.min_cluster_size,
    )
    train_paths = list(izutsumi_paths) + list(other_paths)
    train_labels = [IZUTSUMI_LABEL] * len(izutsumi_paths) + other_labels
    num_classes = max(train_labels) + 1
    logger.info(
        "Pseudo-labels: %d not_izutsumi crops -> %d classes (label 0 = izutsumi, "
        "%d crops total)",
        len(other_paths), num_classes, len(train_paths),
    )

    # --- Validation split (optional: separability drives early stopping)
    val_paths: list[Path] | None = None
    val_labels: list[int] | None = None
    val_izutsumi = val_crops["izutsumi"]
    val_other = val_crops["not_izutsumi"]
    if len(val_izutsumi) >= 2 and len(val_other) >= 1:
        val_paths = list(val_izutsumi) + list(val_other)
        # Val pseudo-labels are irrelevant — only label == 0 marks izutsumi.
        val_labels = [IZUTSUMI_LABEL] * len(val_izutsumi) + [1] * len(val_other)
    else:
        logger.warning(
            "Validation split unusable (izutsumi=%d need >=2, not_izutsumi=%d "
            "need >=1) — training without early stopping",
            len(val_izutsumi), len(val_other),
        )

    # --- Fine-tune from the SAME pretrained init the mining used, so the
    # pseudo-labels and the trained model share one feature space origin.
    model = copy.deepcopy(embedder)
    s2 = settings.training.stage2
    if s2.image_size != embedder.img_size:
        logger.warning(
            "training.stage2.image_size=%d differs from the embedder's canonical "
            "%d — the embedder resolution wins", s2.image_size, embedder.img_size
        )

    model_dir = Path(settings.paths.model_dir)
    history = train(
        model,
        train_paths,
        train_labels,
        val_paths,
        val_labels,
        num_classes=num_classes,
        device=device,
        batch_size=s2.batch_size,
        epochs=s2.epochs,
        learning_rate=s2.learning_rate,
        patience=s2.patience,
        temperature=s2.temperature,
        num_subcenters=s2.num_subcenters,
        model_path=model_dir / s2.model,
    )

    # --- Persist the label->class map for the clustering/graph consumers
    class_map = {
        "izutsumi_label": IZUTSUMI_LABEL,
        "num_classes": num_classes,
        "classes": {
            str(label): (
                "izutsumi"
                if label == IZUTSUMI_LABEL
                else f"not_izutsumi_pseudo_{label}"
            )
            for label in range(num_classes)
        },
    }
    class_map_path = model_dir / CLASS_MAP_FILENAME
    class_map_path.parent.mkdir(parents=True, exist_ok=True)
    class_map_path.write_text(json.dumps(class_map, indent=2, sort_keys=True) + "\n")
    logger.info("Saved class map to %s", class_map_path)

    logger.info(
        "Stage 2 training complete — epochs run: %d, best val separability: %s",
        len(history["train_loss"]),
        ("%.4f" % history["best_val_separability"])
        if history["best_val_separability"] is not None else "n/a",
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
