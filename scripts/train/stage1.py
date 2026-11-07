#!/usr/bin/env python3
"""
Stage 1 YOLO Training Script — body/head/face detection.

Trains the YOLO detection model for manga character parts (body, head,
face) on the SAHI-pre-sliced dataset (train/inference byte parity).
Handles OOM errors with an automatic batch-size ladder.

Stage 1 v2 (docs/implementation_plan.md §4) adds on top of the base flow:
- Primary profile: yolo26m on Colab Pro L4 (config/pipeline.yaml). The
  local M3 chain keeps nano weights via the `train-stage1-m3` pixi task.
- Color-to-line-art bridging: RandomGrayscale(p) with a small brightness/
  contrast jitter, injected into the ultralytics training transform
  pipeline (see _install_random_grayscale).
- Golden polish mode (--polish): a short, low-LR, mosaic-free fine-tuning
  pass over the phase-1 checkpoint with izutsumi slices oversampled 3x,
  written to a separate "<run_name>_polish" run dir but shipped under the
  same target model name.
"""

import argparse
import gc
import logging
import os
import random
import re
import shutil
import sys
from pathlib import Path

# torchvision::nms has no MPS implementation; without this every
# epoch-end validation dies with NotImplementedError. Set before torch
# import so op dispatch sees it. Harmless on cuda/cpu (MPS-only behavior).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.augment import Compose

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cli import _apply_overrides
from src.config import Stage1TrainingParams
from src.training.preparation import save_best_model

logger = logging.getLogger(__name__)

# Batches between mid-epoch torch.mps.empty_cache() calls (see
# _install_mps_cache_guard). 50 ≈ once every ~6 min on the M3 subset run.
MPS_CACHE_CLEAR_INTERVAL = 50

_mps_workers_patched = False


def _install_mps_cache_guard(model: YOLO, device: str) -> None:
    """Bound MPS allocator growth mid-epoch (macOS jetsam protection).

    ultralytics clears the MPS cache between epochs only. On unified-memory
    Macs with foreground apps resident, allocator pool growth within a
    single epoch is fatal: observed 4.3G -> 8.2G driver-allocated over ~650
    iters (batch 16), at which point macOS jetsam killed the process
    ("low-swap", exit 137) on a 16 GB machine (ultralytics#22621 pattern).
    Emptying the cache every N batches keeps growth bounded; the brief
    allocator hitch is amortized over the epoch.
    """
    if str(device).lower() != "mps":
        return

    batches_since_clear = 0
    total_batches = 0

    def clear_mps_cache(trainer) -> None:
        nonlocal batches_since_clear, total_batches
        batches_since_clear += 1
        total_batches += 1
        if batches_since_clear >= MPS_CACHE_CLEAR_INTERVAL:
            gc.collect()
            before = torch.mps.driver_allocated_memory() / 1e9
            torch.mps.empty_cache()
            after = torch.mps.driver_allocated_memory() / 1e9
            logger.info(
                "mps guard @batch %d: gc+empty_cache %.2fG -> %.2fG, torch-allocated %.2fG",
                total_batches, before, after,
                torch.mps.current_allocated_memory() / 1e9,
            )
            batches_since_clear = 0

    model.add_callback("on_train_batch_end", clear_mps_cache)


def _patch_mps_workers(workers: int) -> None:
    """Restore configured dataloader workers on MPS (process-local patch).

    ultralytics BaseTrainer.__init__ forces ``self.args.workers = 0`` for
    cpu/mps devices ("faster CPU training as time dominated by inference").
    That assumption is inverted for yolo26n @640: the nano model's compute
    is tiny and CPU-side augmentation (mosaic, HSV, jpeg decode) dominates —
    the 2.8 s/it baseline on M3 was single-threaded dataloading, not GPU.
    Guarded to MPS and to workers > 0; every other path keeps upstream
    behaviour. Patched once; the OOM retry loop may re-enter train().
    """
    global _mps_workers_patched
    if _mps_workers_patched or workers <= 0:
        return

    from ultralytics.engine import trainer as trainer_module

    original_init = trainer_module.BaseTrainer.__init__

    def init_with_mps_workers(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if self.device.type == "mps" and self.args.workers == 0:
            self.args.workers = workers

    trainer_module.BaseTrainer.__init__ = init_with_mps_workers
    _mps_workers_patched = True


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


def _oom_batch_ladder(batch: int) -> list[int]:
    """Batch sizes to attempt under OOM: the configured batch first, then
    the fixed fallbacks strictly below it (M3 batch 8 still yields
    [8, 4, 2]; the L4 batch 48 degrades 32 -> 16 -> 8 -> 4 -> 2 per plan)."""
    if batch <= 0:
        raise ValueError(f"batch must be positive, got {batch}")
    return [batch] + [b for b in (32, 16, 8, 4, 2) if b < batch]


def _resolve_run_names(stage1: Stage1TrainingParams, polish: bool = False) -> tuple[str, str]:
    """Resolve the (run_name, target_model_name) pair for a training run.

    Configured names (training.stage1.run_name / target_model_name) win;
    empty strings fall back to the historical names keyed on use_subset,
    so the M3 chain (the m3 pixi task pins these fields empty) keeps
    writing to runs/detect/stage1_m3 and producing the v1 artifact names.
    Polish runs append "_polish" to the run dir name and reuse the target
    model name so the deliverable file always holds the best weights.
    """
    run_name = stage1.run_name or ("stage1_m3" if stage1.use_subset else "stage1")
    target_name = stage1.target_model_name or (
        "yolo26_stage1_body_head_face_m3.pt" if stage1.use_subset
        else "yolo26_stage1_body_head_face.pt"
    )
    if polish:
        run_name = f"{run_name}_polish"
    return run_name, target_name


# ---------------------------------------------------------------------------
# Color-to-line-art bridge (RandomGrayscale)
# ---------------------------------------------------------------------------
# Patch-point note (inspected ultralytics 8.4.193, the pinned >=8.4.183,<9
# line): training transforms are built in YOLODataset.build_transforms
# (which calls v8_transforms and appends Format), and the only first-class
# custom hook, hyp.augmentations -> Albumentations, silently no-ops without
# the albumentations package — a forbidden hard dependency here. So the
# bridge wraps YOLODataset.build_transforms the same way _patch_mps_workers
# wraps BaseTrainer.__init__: process-local, patched once, inserting a pure
# cv2/numpy transform right before the trailing Format so val/test datasets
# (built with augment=False) stay identical to the inference pipeline.

# Fixed jitter hyperparams on the grayscaled image: multiplicative
# contrast gain alpha and additive brightness offset beta (uint8 range).
_GRAYSCALE_JITTER_ALPHA_RANGE = (0.85, 1.15)
_GRAYSCALE_JITTER_BETA_RANGE = (-12.0, 12.0)

_random_grayscale_patched = False


def _apply_random_grayscale(img: np.ndarray, p: float, rng) -> np.ndarray:
    """Convert a BGR image to 3-channel grayscale with probability ``p``.

    The converted image gets a small brightness/contrast jitter (contrast
    gain ~ U[0.85, 1.15], brightness offset ~ U[-12, +12] uint8, fixed in
    _GRAYSCALE_JITTER_*_RANGE) so early conv filters see screentone-like
    gray variance instead of flat gray. Pure function: deterministic under
    a seeded ``rng`` (anything with .random()/.uniform(), e.g. the stdlib
    ``random`` module or a ``random.Random`` instance). Non-3-channel
    inputs and p <= 0 are returned unchanged; shape and dtype are
    preserved.
    """
    if p <= 0 or img.ndim != 3 or img.shape[-1] != 3:
        return img
    if rng.random() >= p:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # training pipeline is BGR
    alpha = rng.uniform(*_GRAYSCALE_JITTER_ALPHA_RANGE)
    beta = rng.uniform(*_GRAYSCALE_JITTER_BETA_RANGE)
    jittered = gray.astype(np.float32) * alpha + beta
    gray_u8 = np.clip(np.round(jittered), 0, 255).astype(img.dtype)
    return np.repeat(gray_u8[..., None], 3, axis=-1)


class _RandomGrayscaleTransform:
    """Ultralytics Compose element applying _apply_random_grayscale.

    Duck-types the Compose contract (callable over the labels dict): only
    labels["img"] changes; boxes never move. ``rng=None`` resolves to the
    stdlib ``random`` module, which ultralytics re-seeds per run
    (utils.torch_utils.init_seeds) — grayscale draws then participate in
    the run-level seed, and the instance stays picklable for dataloader
    workers.
    """

    def __init__(self, p: float, rng=None) -> None:
        self.p = float(p)
        self.rng = rng  # None -> stdlib random module (resolved at call time)

    def __call__(self, labels: dict) -> dict:
        img = labels.get("img")
        if not isinstance(img, np.ndarray):
            return labels
        rng = random if self.rng is None else self.rng
        labels["img"] = _apply_random_grayscale(img, self.p, rng)
        return labels


def _insert_random_grayscale(transforms: Compose, p: float, augmenting: bool) -> Compose:
    """Return a new Compose with the grayscale transform placed right before
    the trailing Format entry, or ``transforms`` unchanged when the bridge
    is disabled (p <= 0) or the pipeline is not a train-time one
    (augmenting False — val/test builds keep inference fidelity)."""
    if p <= 0 or not augmenting or not transforms.transforms:
        return transforms
    *body, fmt = transforms.transforms
    return Compose([*body, _RandomGrayscaleTransform(p), fmt])


def _install_random_grayscale(p: float) -> bool:
    """Patch YOLODataset.build_transforms to insert the grayscale bridge.

    Process-local and idempotent (patched-once guard, mirroring
    _patch_mps_workers). Wrapping build_transforms rather than
    v8_transforms covers every build path — including the close_mosaic
    rebuild during the final epochs — and lets the wrapper read
    dataset.augment so validation/test pipelines are untouched. Returns
    True when this call applied the patch, False when disabled (p <= 0)
    or already installed.
    """
    global _random_grayscale_patched
    if _random_grayscale_patched or p <= 0:
        return False

    from ultralytics.data import dataset as dataset_module

    original_build = dataset_module.YOLODataset.build_transforms

    def build_transforms_with_grayscale(dataset_self, hyp):
        transforms = original_build(dataset_self, hyp)
        augmenting = bool(getattr(dataset_self, "augment", False))
        return _insert_random_grayscale(transforms, p, augmenting)

    dataset_module.YOLODataset.build_transforms = build_transforms_with_grayscale
    _random_grayscale_patched = True
    logger.info("RandomGrayscale bridge installed (p=%.2f) before Format.", p)
    return True


# ---------------------------------------------------------------------------
# Golden polish (--polish)
# ---------------------------------------------------------------------------
# docs/implementation_plan.md §4: after the main run converges, refine box
# regression with a short, low-LR, mosaic-free pass on the phase-1 best
# checkpoint, oversampling izutsumi slices 3x. Forced defaults below are
# each overridable via the regular `--override training.stage1.<key>=...`
# mechanism (the forced value is skipped for keys the user overrode).

_POLISH_DEFAULTS: dict[str, object] = {
    "epochs": 10,
    "lr0": 1e-4,
    "mosaic": 0.0,
    "cos_lr": False,
    "close_mosaic": 0,
}
_POLISH_DUP_TAG = "_polishdup"


def _apply_polish_profile(
    stage1: Stage1TrainingParams, overrides: list[str]
) -> Stage1TrainingParams:
    """Return a copy of the stage1 params with polish defaults applied to
    every field the user did not override explicitly on the CLI."""
    overridden = {ov.split("=", 1)[0].strip().lower() for ov in overrides if "=" in ov}
    update = {
        key: value
        for key, value in _POLISH_DEFAULTS.items()
        if f"training.stage1.{key}".lower() not in overridden
    }
    return stage1.model_copy(update=update) if update else stage1


def _rewrite_dataset_yaml_path(yaml_text: str, new_path: str) -> str:
    """Rewrite the ``path:`` line of a write_sliced_dataset_yaml dataset.yaml
    to ``new_path``, leaving every other line byte-identical (comment
    header, train/val/test, names block)."""
    if not re.search(r"(?m)^\s*path\s*:", yaml_text):
        raise ValueError(
            "dataset.yaml lacks a `path:` line (expected the format written "
            "by src/training/slicing.py:write_sliced_dataset_yaml)"
        )
    return re.sub(
        r"(?m)^(\s*)path\s*:\s*.*$",
        lambda match: f"{match.group(1)}path: {new_path}",
        yaml_text,
        count=1,
    )


def _symlink_relative(target: Path, link_path: Path) -> None:
    """Symlink ``link_path`` to ``target`` via a relative path (survives
    moving the whole tree; callers always build a fresh output tree)."""
    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(os.path.relpath(target, link_path.parent))


def _build_polish_dataset(
    dataset_dir: Path,
    output_dir: Path,
    prefix: str = "izutsumi",
    factor: int = 3,
    rebuild: bool = False,
) -> dict:
    """Build a golden-polish copy of the sliced dataset with izutsumi 3x.

    Mirrors the images/{train,val,test} + labels/{train,val,test} +
    dataset.yaml tree produced by scripts/unify/stage1.py --slice /
    src/training/slicing.py. Originals appear in ``output_dir`` as relative
    symlinks (the 14 GB tree must never be copied). The train split
    additionally gains exactly ``factor - 1`` duplicates per slice whose
    stem starts with ``prefix``: the duplicate image is a symlink named
    ``{stem}_polishdup{i}`` (i in 1..factor-1) and its label is a real
    copied file, so a missing original label fails loudly instead of
    producing a duplicate that poisons training semantics. Stems already
    containing ``_polishdup`` are never re-duplicated (feeding a polish
    tree back in stays safe). ``*.cache`` files are skipped: label scans
    fingerprint the source tree and must regenerate under the polish path.
    Refuses to build over an existing ``output_dir`` unless ``rebuild``.
    Deterministic: every listing is sorted.

    Returns a stats dict with per-split image counts, the izutsumi
    counts, and the number of duplicates added.
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    if factor < 1:
        raise ValueError(f"oversampling factor must be >= 1, got {factor}")

    source_yaml = dataset_dir / "dataset.yaml"
    if not source_yaml.exists():
        raise FileNotFoundError(f"sliced dataset.yaml not found at {source_yaml}")

    if output_dir.exists():
        if not rebuild:
            raise FileExistsError(
                f"polish dataset already exists at {output_dir}. Pass "
                "--polish-rebuild to rebuild it; refusing stale reuse."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    stats: dict[str, object] = {
        "output_dir": str(output_dir),
        "factor": factor,
        "train_images": 0,
        "train_originals": 0,
        "izutsumi_train": 0,
        "izutsumi_train_effective": 0,
        "duplicates_added": 0,
        "val_images": 0,
        "test_images": 0,
    }

    for split in ("train", "val", "test"):
        src_images = dataset_dir / "images" / split
        if not src_images.exists():
            continue
        src_labels = dataset_dir / "labels" / split
        if not src_labels.exists():
            raise FileNotFoundError(
                f"labels split missing: expected {src_labels} next to "
                f"{src_images} (sliced datasets always pair both)"
            )
        out_images = output_dir / "images" / split
        out_labels = output_dir / "labels" / split
        out_images.mkdir(parents=True)
        out_labels.mkdir(parents=True)

        # Labels: plain symlinks; stale ultralytics scans (.cache) must
        # not cross trees — they embed absolute paths of the source split.
        for src_label in sorted(src_labels.iterdir()):
            if not src_label.is_file():
                raise ValueError(
                    f"unexpected directory {src_label} inside labels split; "
                    "expected the flat sliced layout"
                )
            if src_label.suffix == ".cache":
                continue
            _symlink_relative(src_label, out_labels / src_label.name)

        # Images: originals symlinked; train izutsumi slices additionally
        # gain (factor - 1) deterministic duplicate entries.
        is_train = split == "train"
        for src_image in sorted(src_images.iterdir()):
            if not src_image.is_file():
                raise ValueError(
                    f"unexpected directory {src_image} inside images split; "
                    "expected the flat sliced layout"
                )
            _symlink_relative(src_image, out_images / src_image.name)
            stats[f"{split}_images"] = int(stats[f"{split}_images"]) + 1
            if is_train:
                stats["train_originals"] = int(stats["train_originals"]) + 1

            stem = src_image.stem
            if not is_train or not stem.startswith(prefix) or _POLISH_DUP_TAG in stem:
                continue

            src_label = src_labels / f"{stem}.txt"
            if not src_label.exists():
                raise FileNotFoundError(
                    f"missing label {src_label} for izutsumi slice "
                    f"{src_image.name}; polish refuses to duplicate "
                    "unlabeled slices"
                )
            for i in range(1, factor):
                _symlink_relative(src_image, out_images / f"{stem}{_POLISH_DUP_TAG}{i}{src_image.suffix}")
                shutil.copy2(src_label, out_labels / f"{stem}{_POLISH_DUP_TAG}{i}.txt")
                stats["duplicates_added"] = int(stats["duplicates_added"]) + 1
                stats["train_images"] = int(stats["train_images"]) + 1
            stats["izutsumi_train"] = int(stats["izutsumi_train"]) + 1

    # Loose root entries (anything the unify pipeline drops beside the
    # splits): mirror, except the yaml (rewritten below) and cache scans.
    for entry in sorted(dataset_dir.iterdir()):
        if entry.name in ("images", "labels", "dataset.yaml") or entry.suffix == ".cache":
            continue
        _symlink_relative(entry, output_dir / entry.name)

    yaml_text = source_yaml.read_text(encoding="utf-8")
    (output_dir / "dataset.yaml").write_text(
        _rewrite_dataset_yaml_path(yaml_text, str(output_dir.resolve())),
        encoding="utf-8",
    )

    stats["izutsumi_train_effective"] = int(stats["izutsumi_train"]) * factor
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Stage 1 YOLO body/head/face detection model.",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("config/pipeline.yaml"),
        help="Path to pipeline config YAML (default: config/pipeline.yaml)",
    )
    parser.add_argument(
        "--dataset", type=Path, default=None,
        help="Explicit path to dataset.yaml or sliced dataset directory.",
    )
    parser.add_argument(
        "--override", action="append", default=[], metavar="KEY=VALUE",
        help="Override config values via dot-notation (e.g. training.stage1.batch=4)",
    )
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Path to checkpoint to resume from (chain resume; or phase-1 best.pt with --polish).",
    )
    parser.add_argument(
        "--polish", action="store_true",
        help="Golden polish: 10 low-LR epochs over --resume weights with izutsumi slices oversampled 3x (into <run>_polish). Requires --resume.",
    )
    parser.add_argument(
        "--polish-rebuild", action="store_true",
        help="Rebuild the polish dataset even if it already exists.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging.",
    )

    args = parser.parse_args()
    _setup_logging(args.verbose)

    # Polish refines an existing phase-1 run; without a checkpoint there is
    # nothing to polish. Fail before any config or model load side effects.
    if args.polish and args.resume is None:
        logger.error(
            "--polish requires --resume <phase1 best.pt> "
            "(runs/detect/<run_name>/weights/best.pt)."
        )
        sys.exit(2)

    # Load settings (exits on failure via _apply_overrides)
    settings = _apply_overrides(args.config, args.override)
    stage1 = settings.training.stage1

    if args.polish:
        stage1 = _apply_polish_profile(stage1, args.override)
        if stage1.use_subset:
            logger.error(
                "--polish oversamples the full sliced dataset; set "
                "training.stage1.use_subset=false."
            )
            sys.exit(2)

    # Warn if the requested GPU/accelerator is not available
    _check_gpu(stage1.device)

    # --- Resolve training dataset ---
    # Priority:
    # 1. Explicit --dataset CLI argument (file or directory).
    # 2. Stratified subset (when stage1.use_subset=true, M3 local profile).
    # 3. Sliced dataset at settings.paths.stage1_sliced_dir (SAHI parity).
    # 4. v2 Sliced dataset at <paths.data>/training/stage1_v2_sliced.
    # 5. Full unified dataset at settings.paths.unified_dir.
    if args.dataset is not None:
        explicit_ds = Path(args.dataset).resolve()
        if explicit_ds.is_dir():
            dataset_yaml = explicit_ds / "dataset.yaml"
            dataset_dir = explicit_ds
        elif explicit_ds.is_file():
            dataset_yaml = explicit_ds
            dataset_dir = explicit_ds.parent
        else:
            logger.error("Explicit --dataset path not found: %s", args.dataset)
            sys.exit(1)

        if not dataset_yaml.exists():
            logger.error("No dataset.yaml found at %s", dataset_yaml)
            sys.exit(1)
    elif stage1.use_subset:
        subset_yaml = settings.paths.stage1_sliced_subset_dir / "dataset.yaml"
        if not subset_yaml.exists():
            logger.error(
                "use_subset=true but %s not found. Run 'pixi run subset-stage1' first.",
                subset_yaml,
            )
            sys.exit(1)
        dataset_yaml = subset_yaml
        dataset_dir = settings.paths.stage1_sliced_subset_dir
    elif (settings.paths.stage1_sliced_dir / "dataset.yaml").exists():
        dataset_yaml = settings.paths.stage1_sliced_dir / "dataset.yaml"
        dataset_dir = settings.paths.stage1_sliced_dir
    else:
        v2_sliced_yaml = settings.paths.stage1_sliced_dir.parent / "stage1_v2_sliced" / "dataset.yaml"
        unified_yaml = settings.paths.unified_dir / "dataset.yaml"
        if v2_sliced_yaml.exists():
            dataset_yaml = v2_sliced_yaml
            dataset_dir = v2_sliced_yaml.parent
            logger.info("Discovered Stage 1 v2 sliced dataset at %s", dataset_yaml)
        elif unified_yaml.exists():
            dataset_yaml = unified_yaml
            dataset_dir = settings.paths.unified_dir
        else:
            logger.error(
                "No dataset.yaml found at %s, %s, or %s. "
                "Run 'python scripts/unify/stage1.py --slice' to generate it.",
                settings.paths.stage1_sliced_dir,
                v2_sliced_yaml.parent,
                settings.paths.unified_dir,
            )
            sys.exit(1)

    # --- Golden polish: build the 3x-oversampled dataset copy ---
    # Polish calibrates box regression on Dungeon Meshi annotations, so it
    # oversamples the full sliced dataset — not the M3-local subset.
    if args.polish:
        if not dataset_yaml.exists():
            logger.error(
                "--polish requires the sliced dataset at %s. Run 'pixi run unify-stage1' first.",
                dataset_yaml,
            )
            sys.exit(1)
        polish_dir = dataset_dir.with_name(dataset_dir.name + "_polish")
        polish_stats = _build_polish_dataset(
            dataset_dir, polish_dir, rebuild=args.polish_rebuild
        )
        logger.info("Polish dataset ready: %s", polish_stats)
        dataset_yaml = polish_dir / "dataset.yaml"

    # --- Determine model source ---
    # Normalize via Path() to collapse double slashes (config has a trailing
    # slash on `data: catnip-data/` which otherwise produces `catnip-data//...`).
    model_path = str(Path(stage1.model))
    if args.resume is not None:
        model_path = str(args.resume)
        if args.polish:
            logger.info("Polish: loading phase-1 weights from: %s", model_path)
        else:
            logger.info("Resuming from checkpoint: %s", model_path)

    logger.info(
        "Starting Stage 1 training: model=%s, device=%s, epochs=%d, batch=%d",
        model_path, stage1.device, stage1.epochs, stage1.batch,
    )
    logger.info("Dataset: %s", dataset_yaml)

    # Color-to-line-art bridge: inject RandomGrayscale before training so
    # every dataset build (train datasets only; see the patch note above)
    # carries the grayscale transform.
    if stage1.random_grayscale > 0:
        _install_random_grayscale(stage1.random_grayscale)

    # Ultralytics clamps dataloader workers to 0 on MPS; restore them so
    # CPU-side augmentation overlaps GPU compute (see _patch_mps_workers).
    if str(stage1.device).lower() == "mps":
        _patch_mps_workers(stage1.workers)

    # --- Resolve run identity ---
    run_name, target_name = _resolve_run_names(stage1, polish=args.polish)
    if args.polish:
        logger.info(
            "Golden polish profile: epochs=%d, lr0=%g, mosaic=%g, cos_lr=%s, close_mosaic=%d",
            stage1.epochs, stage1.lr0, stage1.mosaic, stage1.cos_lr, stage1.close_mosaic,
        )

    # --- Train with OOM retry ---
    batch_attempts = _oom_batch_ladder(stage1.batch)
    trained = False

    for attempt, batch in enumerate(batch_attempts):
        if attempt > 0:
            logger.warning(
                "OOM on batch=%d. Retrying with batch=%d.",
                batch_attempts[attempt - 1], batch,
            )

        try:
            model = YOLO(model_path)
            _install_mps_cache_guard(model, stage1.device)

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
                "close_mosaic": stage1.close_mosaic,
                "lr0": stage1.lr0,
                "cos_lr": stage1.cos_lr,
                "weight_decay": stage1.weight_decay,
                "cls_pw": stage1.cls_pw,
                "box": stage1.box,
                "dfl": stage1.dfl,
                "hsv_h": stage1.hsv_h,
                "hsv_s": stage1.hsv_s,
                "hsv_v": stage1.hsv_v,
                "degrees": stage1.degrees,
                "plots": stage1.plots,
                "save_period": stage1.save_period,
                "project": str(settings.paths.runs_dir / "detect"),
                "name": run_name,
                "exist_ok": True,
                "amp": False if str(stage1.device).lower() == "mps" else True,
                # Explicit optimizer: `auto` ignores manual lr0 (config sets
                # MuSGD for CUDA paths; MPS override below re-forces AdamW).
                "optimizer": stage1.optimizer,
                "deterministic": stage1.deterministic,
            }
            if stage1.mixup > 0:
                train_kwargs["mixup"] = stage1.mixup

            # Force AdamW on MPS to avoid MuSGD bf16 type casting issues
            if str(stage1.device).lower() == "mps":
                train_kwargs["optimizer"] = "AdamW"

            # Chain resume (m3_chain.sh / remote runner): ultralytics
            # restores ALL args + optimizer state from the checkpoint, so
            # mid-resume batch degradation is refused outright. Polish is
            # different: --resume loads phase-1 weights into a fresh short
            # run (<run>_polish), so the OOM ladder stays active there.
            if args.resume is not None and not args.polish:
                if attempt > 0:
                    logger.error(
                        "OOM on batch=%d during resume from %s. Refusing silent fresh restart. Aborting.",
                        batch_attempts[attempt - 1], args.resume,
                    )
                    sys.exit(1)
                train_kwargs["resume"] = True

            model.train(**train_kwargs)
            trained = True
            break

        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            if attempt >= len(batch_attempts) - 1:
                logger.error(
                    "Training failed: OOM even with batch=%d. Exiting.",
                    batch,
                )
                sys.exit(1)
            # Continue to next smaller batch size

    if not trained:
        logger.error("Training did not complete successfully.")
        sys.exit(1)

    # --- Save best model ---
    best_path = save_best_model(
        project_dir=settings.paths.runs_dir / "detect",
        run_name=run_name,
        target_dir=settings.paths.model_dir,
        target_name=target_name,
    )

    if best_path is None:
        logger.error(
            "Failed to save best model. best.pt not found in %s",
            settings.paths.runs_dir / "detect" / run_name / "weights",
        )
        sys.exit(1)

    logger.info("Training complete. Results saved.")


if __name__ == "__main__":
    main()
