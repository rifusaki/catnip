#!/usr/bin/env python3
"""
benchmark_faces.py — Comparative Face and Head Detection Benchmark Harness.

Benchmarks Catnip Stage 1 (YOLO26 Nano) against external state-of-the-art
anime face and head detectors:
  1. Fuyucch1/yolov8_animeface (YOLOv8x6 @ 1280px)
  2. zymk9/yolov5_anime (YOLOv5s @ 640px)
  3. deepghs/anime_face_detection (YOLOv8n @ 640px — direct Nano peer)
  4. deepghs/anime_head_detection (YOLO11n @ 640px — head benchmark)

Usage:
    # Run face detection benchmark on 200 sample test slices
    python scripts/eval/benchmark_faces.py --sample 200 --device mps

    # Download benchmark model weights only
    python scripts/eval/benchmark_faces.py --download-only

    # Run head detection benchmark (defaults automatically to head peers)
    python scripts/eval/benchmark_faces.py --target head
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

# Enable MPS fallback op dispatch for macOS before torch imports
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
if os.environ.get("CUDA_VISIBLE_DEVICES") == "cuda":
    del os.environ["CUDA_VISIBLE_DEVICES"]

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch
import yaml

from src.config import settings
from src.eval.adapters import (
    BENCHMARK_CATALOG,
    BaseDetector,
    CatnipDetector,
    Detection,
    download_benchmark_model,
    get_detector,
)
from src.eval.metrics import EvaluationMetrics, evaluate_detector
from src.eval.visualize import render_comparison_grid

logger = logging.getLogger("benchmark")


# ============================================================================
# CLI Argument Parsing (Law 2 — Make Illegal States Unrepresentable)
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Catnip Comparative Face and Head Detection Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--target",
        choices=["face", "head"],
        default="face",
        help="Target anatomical feature to benchmark",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated list of models to evaluate (defaults dynamically based on --target)",
    )
    parser.add_argument(
        "--catnip-model",
        type=Path,
        default=None,
        help="Path to Catnip Stage 1 checkpoint (.pt)",
    )
    parser.add_argument(
        "--dataset-yaml",
        type=Path,
        default=None,
        help="Path to YOLO dataset.yaml (defaults to stage1_sliced_dir/dataset.yaml)",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Directory of evaluation images (overrides dataset-yaml)",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="Directory of evaluation YOLO .txt labels (overrides dataset-yaml)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Maximum number of test images to evaluate (0 = evaluate all available)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Compute device: 'auto', 'mps', 'cuda', or 'cpu'",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="Confidence detection threshold",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/benchmarks"),
        help="Directory to save benchmark reports, metrics, and visual grids",
    )
    parser.add_argument(
        "--visual-samples",
        type=int,
        default=12,
        help="Number of side-by-side visual comparison panels to generate",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download all benchmark model checkpoints and exit",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible test sampling",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose DEBUG logging",
    )
    return parser.parse_args()


# ============================================================================
# Device Resolution & Setup
# ============================================================================

def resolve_device(device_arg: str) -> str:
    dev = device_arg.lower().strip()
    if dev == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return dev


# ============================================================================
# Ground Truth Loading & Label Parsing (Law 1 — Early Exit, Law 4 — Fail Fast)
# ============================================================================

def parse_yolo_label(label_path: Path, target_gt_class: int) -> list[Detection]:
    """Parse YOLO txt format label and extract target class as normalized Detection."""
    if not label_path.exists() or label_path.stat().st_size == 0:
        return []

    boxes: list[Detection] = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            try:
                cls_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])
            except ValueError:
                continue

            if cls_id != target_gt_class:
                continue

            # Convert center-xywh to xyxy
            x1 = max(0.0, xc - w / 2.0)
            y1 = max(0.0, yc - h / 2.0)
            x2 = min(1.0, xc + w / 2.0)
            y2 = min(1.0, yc + h / 2.0)

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    score=1.0,
                    class_id=0,
                )
            )

    return boxes


def resolve_eval_dataset(
    args: argparse.Namespace,
) -> tuple[list[tuple[Path, Path]], int]:
    """
    Resolve test image and label paths and determine the target GT class index.
    Returns: (list of (image_path, label_path), target_gt_class_id)
    """
    target_gt_class = CatnipDetector.CATNIP_CLASSES[args.target]

    images_dir: Path | None = args.images_dir
    labels_dir: Path | None = args.labels_dir

    if images_dir is None:
        yaml_path = args.dataset_yaml or (settings.paths.stage1_sliced_dir / "dataset.yaml")
        if yaml_path.exists():
            with open(yaml_path, "r", encoding="utf-8") as f:
                data_info = yaml.safe_load(f)

            root = Path(data_info.get("path", yaml_path.parent))
            test_rel = data_info.get("test", "images/test")
            images_dir = root / test_rel if not Path(test_rel).is_absolute() else Path(test_rel)
            if labels_dir is None:
                labels_dir = root / "labels" / images_dir.name
        else:
            images_dir = settings.paths.stage1_sliced_dir / "images" / "test"
            if labels_dir is None:
                labels_dir = settings.paths.stage1_sliced_dir / "labels" / "test"

    if not images_dir.exists():
        logger.error("Images directory not found: %s", images_dir)
        logger.info(
            "If running locally without the external drive attached, provide a test directory with "
            "--images-dir <path> --labels-dir <path>"
        )
        sys.exit(1)

    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    all_images = [p for p in images_dir.iterdir() if p.suffix.lower() in image_exts]

    if not all_images:
        logger.error("No test images found in %s", images_dir)
        sys.exit(1)

    pairs: list[tuple[Path, Path]] = []
    missing_labels_count = 0
    for img_path in sorted(all_images):
        if labels_dir is not None:
            lbl_path = labels_dir / f"{img_path.stem}.txt"
        else:
            # Ultralytics convention: replace '/images/' with '/labels/'
            lbl_path = Path(str(img_path).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")).with_suffix(".txt")

        if not lbl_path.exists():
            missing_labels_count += 1
        pairs.append((img_path, lbl_path))

    if missing_labels_count > 0:
        logger.warning(
            "%d / %d image files do not have matching label files (%s)",
            missing_labels_count,
            len(pairs),
            "will be treated as zero ground-truth backgrounds",
        )

    return pairs, target_gt_class


def classify_domain(path: Path) -> str:
    """Classify an evaluation slice into Manga Line-Art vs Anime Illustration."""
    name_lower = str(path).lower()
    if "manga" in name_lower or "izutsumi" in name_lower:
        return "manga_lineart"
    if any(prefix in name_lower for prefix in ("ahv", "ahaf", "ah_", "ah-", "anime", "heads")):
        return "anime_illustration"
    return "general"


# ============================================================================
# Main Benchmark Pipeline
# ============================================================================

def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # 1. Download-only shortcut
    if args.download_only:
        logger.info("Downloading all catalog benchmark models...")
        for key in BENCHMARK_CATALOG:
            download_benchmark_model(key)
        logger.info("All benchmark models cached successfully.")
        return

    # 2. Resolve compute device
    device = resolve_device(args.device)
    logger.info("Benchmarking target '%s' on compute device '%s'", args.target, device)

    # 3. Model list resolution (dynamically driven by --target to prevent target mismatch)
    if not args.models or args.models in ("auto", "all"):
        if args.target == "face":
            model_keys = ["catnip", "fuyucchi_face", "zymk9_face", "deepghs_face_nano"]
        else:
            model_keys = ["catnip", "deepghs_head_nano"]
    else:
        model_keys = [k.strip() for k in args.models.split(",") if k.strip()]

    # 4. Initialize detectors
    detectors: dict[str, BaseDetector] = {}
    for key in model_keys:
        try:
            detector = get_detector(
                name=key,
                target=args.target,
                custom_path=args.catnip_model if key.startswith("catnip") else None,
                device=device,
            )
            # Use unique key to prevent collisions
            detectors[key] = detector
        except Exception as err:
            logger.warning("Could not initialize detector '%s': %s (skipping)", key, err)

    if not detectors:
        logger.error("No detectors were successfully initialized. Exiting.")
        sys.exit(1)

    # 5. Resolve dataset
    image_label_pairs, target_gt_class = resolve_eval_dataset(args)
    logger.info("Found %d test samples (target GT class index: %d)", len(image_label_pairs), target_gt_class)

    # Sampling
    random.seed(args.seed)
    if args.sample > 0 and args.sample < len(image_label_pairs):
        random.shuffle(image_label_pairs)
        image_label_pairs = image_label_pairs[:args.sample]
        logger.info("Sampled %d images for evaluation", len(image_label_pairs))

    # 6. Run Inference & Metric Collection
    ground_truths: dict[str, list[Detection]] = {}
    domain_by_img: dict[str, str] = {}
    predictions: dict[str, dict[str, list[Detection]]] = {key: {} for key in detectors}
    latencies: dict[str, dict[str, list[float]]] = {key: defaultdict(list) for key in detectors}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir = args.output_dir / "visuals"
    visual_candidates: list[tuple[float, Path, list[Detection], dict[str, list[Detection]]]] = []

    logger.info("Running evaluation across %d test images...", len(image_label_pairs))
    total_imgs = len(image_label_pairs)
    skipped_images = 0

    for idx, (img_path, lbl_path) in enumerate(image_label_pairs):
        # Unique identifier using relative path or filename
        img_id = f"{img_path.parent.name}_{img_path.name}"
        domain = classify_domain(img_path)

        # Law 1 & Law 4: Verify image loads before registering GTs (prevents phantom misses)
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning("Failed to read image %s. Skipping without recording ground truth.", img_path)
            skipped_images += 1
            continue

        gt_boxes = parse_yolo_label(lbl_path, target_gt_class=target_gt_class)
        ground_truths[img_id] = gt_boxes
        domain_by_img[img_id] = domain

        preds_this_img: dict[str, list[Detection]] = {}

        for key, detector in detectors.items():
            t0 = time.perf_counter()
            preds = detector.predict(image, conf=args.conf)
            dt_ms = (time.perf_counter() - t0) * 1000.0

            latencies[key]["overall"].append(dt_ms)
            latencies[key][domain].append(dt_ms)
            predictions[key][img_id] = preds
            preds_this_img[detector.name] = preds

        # Check for disagreement / interesting visual comparison cases
        pred_counts = [len(p) for p in preds_this_img.values()]
        disagreement_score = float(np.std(pred_counts) + (1.0 if len(gt_boxes) > 0 else 0.0))
        visual_candidates.append((disagreement_score, img_path, gt_boxes, preds_this_img))

        if (idx + 1) % 25 == 0 or (idx + 1) == total_imgs:
            logger.info("Processed %d / %d images (%.1f%%)", idx + 1, total_imgs, ((idx + 1) / total_imgs) * 100)

    evaluated_count = len(ground_truths)
    logger.info("Evaluation complete: %d evaluated images, %d skipped due to read errors.", evaluated_count, skipped_images)

    # 7. Generate Visual Comparison Panels
    # Prioritize panels that contain actual ground truth annotations (gt_boxes > 0)
    # and exhibit meaningful model divergence.
    gt_positive_candidates = [c for c in visual_candidates if len(c[2]) > 0]
    selection_pool = gt_positive_candidates if gt_positive_candidates else visual_candidates
    selection_pool.sort(key=lambda x: x[0], reverse=True)
    n_visuals = min(args.visual_samples, len(selection_pool))
    logger.info("Rendering %d side-by-side visual comparison panels...", n_visuals)

    for v_idx in range(n_visuals):
        _, img_path, gt_boxes, preds_this_img = selection_pool[v_idx]
        image = cv2.imread(str(img_path))
        if image is not None:
            render_comparison_grid(
                image=image,
                ground_truth_boxes=gt_boxes,
                predictions_by_model=preds_this_img,
                output_path=visuals_dir / f"compare_{img_path.stem}.jpg",
            )

    # 8. Compute Metrics (Overall and Stratified per Domain)
    results_all: list[EvaluationMetrics] = []
    results_by_domain: dict[str, list[EvaluationMetrics]] = defaultdict(list)

    domains = ["overall", "manga_lineart", "anime_illustration"]

    for dom in domains:
        if dom == "overall":
            sub_gt = ground_truths
        else:
            sub_gt = {k: v for k, v in ground_truths.items() if domain_by_img.get(k) == dom}

        if not sub_gt:
            continue

        for key, detector in detectors.items():
            sub_preds = {k: predictions[key].get(k, []) for k in sub_gt}
            dom_latencies = latencies[key][dom]
            avg_lat = float(np.mean(dom_latencies)) if dom_latencies else 0.0

            metrics = evaluate_detector(
                model_name=detector.name,
                target_class=args.target,
                domain=dom,
                predictions_by_img=sub_preds,
                ground_truths_by_img=sub_gt,
                avg_latency_ms=avg_lat,
                params_millions=detector.parameter_count / 1e6,
                weight_size_mb=detector.weight_size_bytes / 1e6,
            )

            if dom == "overall":
                results_all.append(metrics)
            results_by_domain[dom].append(metrics)

    # 9. Format Markdown & Terminal Output
    markdown_lines: list[str] = [
        f"# Catnip Comparative Benchmark: {args.target.capitalize()} Detection",
        f"- **Date & Time:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Device:** `{device}`",
        f"- **Evaluated Images:** {evaluated_count} (GT Total: {sum(len(b) for b in ground_truths.values())} boxes, {skipped_images} skipped)",
        f"- **Confidence Threshold:** `{args.conf}`",
        "",
        "## Overall Benchmark Metrics",
        "",
        "| Model | Params | Size | mAP@50 | mAP@50-95 | Precision | Recall | F1 | Latency |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for m in results_all:
        markdown_lines.append(m.to_markdown_row())

    # Domain Stratification Breakdown
    for dom in ("manga_lineart", "anime_illustration"):
        if dom in results_by_domain:
            dom_title = "Manga Line-Art" if dom == "manga_lineart" else "Anime Illustrations"
            markdown_lines.extend([
                "",
                f"### Domain Breakdown: {dom_title}",
                "",
                "| Model | Params | Size | mAP@50 | mAP@50-95 | Precision | Recall | F1 | Latency |",
                "|---|---|---|---|---|---|---|---|---|",
            ])
            for m in results_by_domain[dom]:
                markdown_lines.append(m.to_markdown_row())

    report_text = "\n".join(markdown_lines)
    report_file = args.output_dir / "summary.md"
    report_file.write_text(report_text, encoding="utf-8")

    # JSON output
    json_payload = {
        "target": args.target,
        "device": device,
        "total_images": evaluated_count,
        "skipped_images": skipped_images,
        "metrics_overall": [m.to_dict() for m in results_all],
        "metrics_by_domain": {dom: [m.to_dict() for m in ml] for dom, ml in results_by_domain.items()},
    }
    json_file = args.output_dir / "metrics.json"
    json_file.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    # Print to console
    print("\n" + "=" * 80)
    print(report_text)
    print("=" * 80)
    logger.info("Saved benchmark summary report to %s", report_file)
    logger.info("Saved JSON metrics to %s", json_file)
    logger.info("Saved %d visual comparison panels to %s", n_visuals, visuals_dir)


if __name__ == "__main__":
    main()
