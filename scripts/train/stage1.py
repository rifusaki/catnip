#!/usr/bin/env python3
"""
Stage 1 YOLO Training Script — face/body detection model.

Trains a YOLO model for manga character part detection (face vs body).
Handles OOM errors with automatic batch-size reduction.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cli import _apply_overrides
from src.training.preparation import save_best_model

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging with levels and formatting."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)-8s %(name)-20s %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )


def _check_gpu(device: str) -> None:
    """Log a warning if the requested compute device is unavailable."""
    device_lower = device.lower()
    if device_lower in ("cuda", "cuda:0") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
    elif device_lower == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS requested but not available. Falling back to CPU.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Stage 1 YOLO face/body detection model.",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("config/pipeline.yaml"),
        help="Path to pipeline config YAML (default: config/pipeline.yaml)",
    )
    parser.add_argument(
        "--override", action="append", default=[], metavar="KEY=VALUE",
        help="Override config values via dot-notation (e.g. training.stage1.batch=4)",
    )
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Path to checkpoint to resume training from.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging.",
    )

    args = parser.parse_args()
    _setup_logging(args.verbose)

    # Load settings (exits on failure via _apply_overrides)
    settings = _apply_overrides(args.config, args.override)
    stage1 = settings.training.stage1

    # Warn if the requested GPU/accelerator is not available
    _check_gpu(stage1.device)

    # --- Validate pre-unified training data ---
    # Prefer the SAHI-sliced dataset when available (train/inference parity).
    # Fall back to the unified (full-image) dataset otherwise.
    sliced_yaml = settings.paths.stage1_sliced_dir / "dataset.yaml"
    unified_yaml = settings.paths.unified_dir / "dataset.yaml"
    if sliced_yaml.exists():
        dataset_yaml = sliced_yaml
    elif unified_yaml.exists():
        dataset_yaml = unified_yaml
    else:
        logger.error(
            "No dataset.yaml found at %s or %s. "
            "Run 'python scripts/unify/stage1.py --slice' to generate it.",
            settings.paths.stage1_sliced_dir, settings.paths.unified_dir,
        )
        sys.exit(1)

    # --- Determine model source ---
    # Normalize via Path() to collapse double slashes (config has a trailing
    # slash on `data: catnip-data/` which otherwise produces `catnip-data//...`).
    model_path = str(Path(stage1.model))
    if args.resume is not None:
        model_path = str(args.resume)
        logger.info("Resuming from checkpoint: %s", model_path)

    logger.info(
        "Starting Stage 1 training: model=%s, device=%s, epochs=%d, batch=%d",
        model_path, stage1.device, stage1.epochs, stage1.batch,
    )

    # --- Train with OOM retry ---
    batch_attempts = [stage1.batch, 4, 2]
    trained = False

    for attempt, batch in enumerate(batch_attempts):
        if attempt > 0:
            logger.warning(
                "OOM on batch=%d. Retrying with batch=%d.",
                batch_attempts[attempt - 1], batch,
            )

        try:
            model = YOLO(model_path)

            train_kwargs: dict = {
                "data": str(dataset_yaml),
                "epochs": stage1.epochs,
                "imgsz": stage1.imgsz,
                "batch": batch,
                "workers": stage1.workers,
                "device": stage1.device,
                "patience": stage1.patience,
                "freeze": stage1.freeze,
                "mosaic": stage1.mosaic,
                "lr0": stage1.lr0,
                "cos_lr": stage1.cos_lr,
                "project": str(settings.paths.runs_dir / "detect"),
                "name": "stage1",
                "amp": False if str(stage1.device).lower() == "mps" else True,
            }
            
            # Force AdamW on MPS to avoid MuSGD bf16 type casting issues
            if str(stage1.device).lower() == "mps":
                train_kwargs["optimizer"] = "AdamW"

            # Only attempt resume on the first try with the original batch size.
            # Changing batch size invalidates optimizer state, so OOM retries
            # start fresh with a smaller batch.
            if args.resume is not None and attempt == 0:
                train_kwargs["resume"] = True

            model.train(**train_kwargs)
            trained = True
            break

        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            if attempt >= len(batch_attempts) - 1:
                logger.error(
                    "Training failed: OOM even with batch=%d. Exiting.", batch,
                )
                sys.exit(1)
            # Continue to next smaller batch size

    if not trained:
        logger.error("Training did not complete successfully.")
        sys.exit(1)

    # --- Save best model ---
    best_path = save_best_model(
        project_dir=settings.paths.runs_dir / "detect",
        run_name="stage1",
        target_dir=settings.paths.model_dir,
        target_name="yolo26_stage1_body_head_face.pt",
    )

    if best_path is None:
        logger.error(
            "Failed to save best model. best.pt not found in %s",
            settings.paths.runs_dir / "detect" / "stage1" / "weights",
        )
        sys.exit(1)

    logger.info("Training complete. Results saved.")


if __name__ == "__main__":
    main()
