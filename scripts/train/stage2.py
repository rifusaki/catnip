#!/usr/bin/env python3
"""
Stage 2 Triplet-Loss ReID Training (Placeholder).

Prepares the triplet dataset from YOLO crops and prints the training
configuration.  The actual ResNet18 + TripletMarginLoss training loop is
**not yet implemented** — this script serves as a data-prep step that
validates the pipeline configuration and dataset readiness.

Usage::

    python scripts/train/stage2.py
    python scripts/train/stage2.py --config config/pipeline.yaml
    python scripts/train/stage2.py --verbose
    python scripts/train/stage2.py --override training.stage2.epochs=200
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf

# Allow running the script directly from scripts/train/ without an installed package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (placeholder — real values wired once ReID model is implemented)
# ---------------------------------------------------------------------------

DEFAULT_EMBEDDING_DIM = 128   # ResNet18 output → 128-d embedding
DEFAULT_OPTIMIZER = "adam"


# ---------------------------------------------------------------------------
# Helpers
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
    """Print the Stage 2 training configuration derived from settings.

    Pulls parameters from ``settings.training.stage2`` and augments with
    values from ``settings.params.metric_learning`` where applicable.
    """
    s2 = settings.training.stage2
    ml = settings.params.metric_learning

    logger.info("=== Stage 2 Triplet-Loss ReID Training ===")
    logger.info("  model:           %s", s2.model)
    logger.info("  batch_size:      %d", s2.batch_size)
    logger.info("  epochs:          %d", s2.epochs)
    logger.info("  learning_rate:   %.6f", s2.learning_rate)
    logger.info("  optimizer:       %s", DEFAULT_OPTIMIZER)
    logger.info("  device:          %s", s2.device)
    logger.info("  margin:          %.4f", s2.margin)
    logger.info("  embedding_dim:   %d", DEFAULT_EMBEDDING_DIM)
    logger.info("  img_size:        %d", ml.img_size)
    logger.info("  p (per class):   %d", s2.p)
    logger.info("  k (samples):     %d", s2.k)
    logger.info("  patience:        %d", s2.patience)
    logger.info("")


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
        # Still call for the side-effect of scanning other source dirs,
        # but pass only the (non-existent) dir so the function handles it.
        return {
            "train": {"izutsumi": 0, "not_izutsumi": 0},
            "val": {"izutsumi": 0, "not_izutsumi": 0},
        }

    from src.training.preparation import prepare_triplet_dataset

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


def _log_placeholder_message(settings: Settings) -> None:
    """Emit the required placeholder warnings about not-yet-implemented modules."""
    logger.warning("Stage 2 ReID model training not yet implemented.")
    logger.info(
        "Required components: src/reid/model.py (ResNet18 backbone), "
        "src/reid/dataset.py (TripletDataset), "
        "src/reid/loss.py (TripletMarginLoss)"
    )
    logger.info("Triplet dataset is ready at %s", settings.paths.stage2_dir)
    logger.info(
        "To implement: build embedding extraction with pytorch, "
        "FAISS index for matching"
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2 Triplet-Loss ReID Training (placeholder — data prep only).",
    )
    parser.add_argument(
        "--config",
        default="config/pipeline.yaml",
        help="Path to pipeline configuration YAML (default: %(default)s).",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=None,
        metavar="KEY=VALUE",
        help="Override config values using dot-path keys "
             "(e.g. training.stage2.epochs=200).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    args = parser.parse_args()

    # --- Setup logging ---
    _configure_logging(args.verbose)

    # --- Load and validate configuration (Law 4 — Fail Fast) ---
    settings = _load_and_override_config(args.config, args.override)

    # --- Log training hyperparameters ---
    _log_training_config(settings)

    # --- Prepare triplet dataset ---
    counts = _prepare_dataset(settings)

    # --- Log dataset counts ---
    _log_dataset_counts(counts)

    # --- Placeholder: real training not yet implemented ---
    _log_placeholder_message(settings)

    # Exit 0 — this is expected behaviour for the data-prep placeholder
    sys.exit(0)


if __name__ == "__main__":
    main()
