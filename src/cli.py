#!/usr/bin/env python3
"""
Catnip CLI — Character Re-Identification in manga.
"""

import argparse
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf

from src.config import Settings

logger = logging.getLogger("catnip")


def _apply_overrides(config_path: Path, overrides: list[str]) -> Settings:
    """Merge dot-notation KEY=VALUE overrides into OmegaConf, then re-validate with Pydantic."""
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    cfg = OmegaConf.load(config_path)
    valid_keys = set(Settings.model_fields.keys())

    for override in overrides:
        if "=" not in override:
            logger.error(f"Invalid override format: '{override}'. Expected KEY=VALUE (dot-notation).")
            sys.exit(1)
        key, value = override.split("=", 1)

        root_key = key.split(".")[0]
        if root_key not in valid_keys:
            logger.error(f"Invalid override key '{key}': no top-level key '{root_key}'")
            logger.error(f"Available top-level keys: {', '.join(sorted(valid_keys))}")
            sys.exit(1)

        try:
            if value.lower() in ("true", "false"):
                typed_value = value.lower() == "true"
            else:
                try:
                    typed_value = int(value)
                except ValueError:
                    try:
                        typed_value = float(value)
                    except ValueError:
                        typed_value = value
        except Exception:
            typed_value = value

        try:
            OmegaConf.update(cfg, key, typed_value)
        except Exception as e:
            logger.error(f"Invalid override key '{key}': {e}")
            logger.error(f"Available top-level keys: {', '.join(sorted(valid_keys))}")
            sys.exit(1)

    try:
        resolved = OmegaConf.to_container(cfg, resolve=True)
        return Settings(**resolved)
    except Exception as e:
        logger.error(f"Invalid value for override: {e}")
        sys.exit(1)


def _setup_logging(verbose: bool = False):
    """Configure logging with tqdm-compatible formatting."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)-8s %(name)-20s %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )


def cmd_extract(args):
    """Stage 1: SAHI face/body detection on image directory."""
    settings = _apply_overrides(args.config, args.override)
    _setup_logging(args.verbose)

    logger.info("Extract: SAHI sliced inference (Stage 1)")
    logger.info(f"  Input:  {args.input_dir}")
    logger.info(f"  Output: {args.output_dir or settings.paths.crops_dir}")
    logger.info(f"  Model:  {args.model or settings.paths.model_dir / 'yolo26_stage1_face_body.pt'}")

    logger.error("cmd_extract: Not yet implemented — inference pipeline is planned for after model training is complete.")
    sys.exit(1)


def cmd_embed(args):
    """Stage 2: Compute embeddings for crops."""
    settings = _apply_overrides(args.config, args.override)
    _setup_logging(args.verbose)

    logger.info("Embed: Computing ReID embeddings (Stage 2)")
    logger.info(f"  Input: {args.input_dir}")
    logger.info(f"  Model: {args.model or settings.paths.model_dir / 'reid_resnet18.pth'}")

    logger.error("cmd_embed: Not yet implemented — inference pipeline is planned for after model training is complete.")
    sys.exit(1)


def cmd_match(args):
    """Compare embeddings against Izutsumi seed index."""
    settings = _apply_overrides(args.config, args.override)
    _setup_logging(args.verbose)

    logger.info("Match: Cosine similarity against seed embeddings")
    logger.info(f"  Embeddings: {args.embeddings}")
    logger.info(f"  Threshold:  {args.threshold}")

    logger.error("cmd_match: Not yet implemented — inference pipeline is planned for after model training is complete.")
    sys.exit(1)


def cmd_pipeline(args):
    """Full pipeline: extract → embed → match."""
    settings = _apply_overrides(args.config, args.override)
    _setup_logging(args.verbose)

    logger.info("Pipeline: extract → embed → match")
    logger.info(f"  Input: {args.input_dir}")
    logger.info(f"  Output: {args.output_dir or settings.paths.output_dir}")

    logger.error("cmd_pipeline: Not yet implemented — inference pipeline is planned for after model training is complete.")
    sys.exit(1)


def cmd_train_stage1(args):
    """Train Stage 1 YOLO26 model."""
    settings = _apply_overrides(args.config, args.override)
    _setup_logging(args.verbose)

    logger.info("Train Stage 1: YOLO26 face/body detection")
    logger.info(f"  Model: {settings.training.stage1.model}")
    logger.info(f"  IMGSZ: {settings.training.stage1.imgsz}")
    logger.info(f"  Epochs: {settings.training.stage1.epochs}")
    logger.info(f"  Batch: {settings.training.stage1.batch}")
    logger.info(f"  Device: {settings.training.stage1.device}")

    import subprocess
    script = Path(__file__).parent.parent / "scripts" / "train_stage1.py"
    cmd = [sys.executable, str(script)]
    if args.override:
        for ov in args.override:
            cmd.extend(["--override", ov])
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Train Stage 1 script failed with exit code {e.returncode}")
        sys.exit(e.returncode)


def cmd_train_stage2(args):
    """Train Stage 2 ReID model."""
    settings = _apply_overrides(args.config, args.override)
    _setup_logging(args.verbose)

    logger.info("Train Stage 2: Triplet Loss ReID")
    logger.info(f"  Model: {settings.training.stage2.model}")
    logger.info(f"  Batch Size: {settings.training.stage2.batch_size}")
    logger.info(f"  Epochs: {settings.training.stage2.epochs}")
    logger.info(f"  Learning Rate: {settings.training.stage2.learning_rate}")
    logger.info(f"  Device: {settings.training.stage2.device}")

    import subprocess
    script = Path(__file__).parent.parent / "scripts" / "train_stage2.py"
    cmd = [sys.executable, str(script)]
    if args.override:
        for ov in args.override:
            cmd.extend(["--override", ov])
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Train Stage 2 script failed with exit code {e.returncode}")
        sys.exit(e.returncode)


def main():
    parser = argparse.ArgumentParser(
        prog="catnip",
        description="Character Re-Identification in manga — detect and identify Izutsumi.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging")
    parser.add_argument("--config", type=Path, default="config/pipeline.yaml", help="Path to config file")
    parser.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Override config values using dot-notation (e.g. params.sahi.confidence_threshold=0.5)")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # extract
    p_extract = subparsers.add_parser("extract", help="Stage 1 SAHI face/body detection")
    p_extract.add_argument("--input-dir", type=Path, required=True, help="Directory of manga page images")
    p_extract.add_argument("--output-dir", type=Path, help="Output directory for crops (default: from config)")
    p_extract.add_argument("--model", type=Path, help="Path to YOLO26 model (default: from config)")
    p_extract.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging")
    p_extract.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
    p_extract.set_defaults(func=cmd_extract)

    # embed
    p_embed = subparsers.add_parser("embed", help="Stage 2 ReID embedding computation")
    p_embed.add_argument("--input-dir", type=Path, required=True, help="Directory of character crops")
    p_embed.add_argument("--output", type=Path, help="Output .npy file for embeddings")
    p_embed.add_argument("--model", type=Path, help="Path to ReID model (default: from config)")
    p_embed.add_argument("--verbose", "-v", action="store_true")
    p_embed.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
    p_embed.set_defaults(func=cmd_embed)

    # match
    p_match = subparsers.add_parser("match", help="Match embeddings against Izutsumi seeds")
    p_match.add_argument("--embeddings", type=Path, required=True, help="Path to embeddings .npy file")
    p_match.add_argument("--threshold", type=float, default=0.7, help="Similarity threshold (default: 0.7)")
    p_match.add_argument("--output", type=Path, help="Output JSON path for matches")
    p_match.add_argument("--verbose", "-v", action="store_true")
    p_match.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
    p_match.set_defaults(func=cmd_match)

    # pipeline
    p_pipeline = subparsers.add_parser("pipeline", help="Full pipeline: extract → embed → match")
    p_pipeline.add_argument("--input-dir", type=Path, required=True, help="Directory of manga page images")
    p_pipeline.add_argument("--output-dir", type=Path, help="Output directory for results")
    p_pipeline.add_argument("--verbose", "-v", action="store_true")
    p_pipeline.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
    p_pipeline.set_defaults(func=cmd_pipeline)

    # train stage1
    p_train1 = subparsers.add_parser("train", help="Training commands")
    train_subs = p_train1.add_subparsers(dest="train_command")
    p_t1 = train_subs.add_parser("stage1", help="Train Stage 1 YOLO26 face/body detection")
    p_t1.add_argument("--verbose", "-v", action="store_true")
    p_t1.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
    p_t1.set_defaults(func=cmd_train_stage1)

    p_t2 = train_subs.add_parser("stage2", help="Train Stage 2 Triplet Loss ReID")
    p_t2.add_argument("--verbose", "-v", action="store_true")
    p_t2.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
    p_t2.set_defaults(func=cmd_train_stage2)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
