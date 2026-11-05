"""Unit tests for the Catnip comparative evaluation and benchmarking suite."""

from pathlib import Path
import numpy as np
import pytest

from scripts.eval.benchmark_faces import classify_domain, parse_yolo_label
from src.eval.adapters import (
    BENCHMARK_CATALOG,
    CatnipDetector,
    Detection,
    get_detector,
)
from src.eval.metrics import (
    compute_ap_from_pr,
    compute_iou,
    compute_iou_matrix,
    evaluate_detector,
    evaluate_single_iou,
)


def test_detection_dataclass_valid() -> None:
    det = Detection(x1=0.1, y1=0.2, x2=0.5, y2=0.6, score=0.85, class_id=0)
    assert det.x1 == 0.1
    assert det.score == 0.85
    assert det.class_id == 0

    px1, py1, px2, py2 = det.to_pixel_coords(width=1000, height=2000)
    assert px1 == 100
    assert py1 == 400
    assert px2 == 500
    assert py2 == 1200


def test_detection_dataclass_invalid() -> None:
    # x1 > x2
    with pytest.raises(ValueError, match="Invalid bounding box coordinates"):
        Detection(x1=0.8, y1=0.2, x2=0.4, y2=0.6, score=0.9, class_id=0)

    # score > 1.0
    with pytest.raises(ValueError, match="Confidence score"):
        Detection(x1=0.1, y1=0.2, x2=0.5, y2=0.6, score=1.5, class_id=0)


def test_compute_iou() -> None:
    box1 = Detection(x1=0.0, y1=0.0, x2=0.5, y2=0.5, score=1.0, class_id=0)
    # Identical box -> IoU = 1.0
    assert compute_iou(box1, box1) == pytest.approx(1.0)

    # Non-overlapping box -> IoU = 0.0
    box_disjoint = Detection(x1=0.6, y1=0.6, x2=1.0, y2=1.0, score=1.0, class_id=0)
    assert compute_iou(box1, box_disjoint) == pytest.approx(0.0)

    # Partial overlap
    box_half = Detection(x1=0.0, y1=0.25, x2=0.5, y2=0.5, score=1.0, class_id=0)
    assert compute_iou(box1, box_half) == pytest.approx(0.5)


def test_compute_iou_matrix() -> None:
    preds = [
        Detection(x1=0.0, y1=0.0, x2=0.4, y2=0.4, score=0.9, class_id=0),
        Detection(x1=0.5, y1=0.5, x2=0.9, y2=0.9, score=0.8, class_id=0),
    ]
    gts = [
        Detection(x1=0.0, y1=0.0, x2=0.4, y2=0.4, score=1.0, class_id=0),
        Detection(x1=0.5, y1=0.5, x2=0.9, y2=0.9, score=1.0, class_id=0),
    ]
    matrix = compute_iou_matrix(preds, gts)
    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == pytest.approx(1.0)
    assert matrix[0, 1] == pytest.approx(0.0)
    assert matrix[1, 0] == pytest.approx(0.0)
    assert matrix[1, 1] == pytest.approx(1.0)


def test_compute_ap_from_pr() -> None:
    # Perfect step function
    recalls = np.array([0.2, 0.5, 1.0])
    precisions = np.array([1.0, 1.0, 1.0])
    assert compute_ap_from_pr(recalls, precisions) == pytest.approx(1.0)

    # Linear drop
    recalls = np.array([0.5, 1.0])
    precisions = np.array([1.0, 0.5])
    assert compute_ap_from_pr(recalls, precisions) == pytest.approx(0.75)


def test_evaluate_single_iou_metrics() -> None:
    # Case: 1 image, 2 ground truth boxes, 2 exact predictions
    gts = {
        "img1": [
            Detection(x1=0.1, y1=0.1, x2=0.3, y2=0.3, score=1.0, class_id=0),
            Detection(x1=0.6, y1=0.6, x2=0.8, y2=0.8, score=1.0, class_id=0),
        ]
    }
    preds = {
        "img1": [
            Detection(x1=0.1, y1=0.1, x2=0.3, y2=0.3, score=0.95, class_id=0),
            Detection(x1=0.6, y1=0.6, x2=0.8, y2=0.8, score=0.90, class_id=0),
        ]
    }

    prec, rec, f1, ap = evaluate_single_iou(preds, gts, iou_threshold=0.5)
    assert prec == pytest.approx(1.0)
    assert rec == pytest.approx(1.0)
    assert f1 == pytest.approx(1.0)
    assert ap == pytest.approx(1.0)


def test_evaluate_single_iou_duplicate_rejection() -> None:
    # 1 GT box, 2 overlapping predictions -> second one is FP
    gts = {
        "img1": [
            Detection(x1=0.1, y1=0.1, x2=0.3, y2=0.3, score=1.0, class_id=0),
        ]
    }
    preds = {
        "img1": [
            Detection(x1=0.1, y1=0.1, x2=0.3, y2=0.3, score=0.95, class_id=0),
            Detection(x1=0.12, y1=0.12, x2=0.32, y2=0.32, score=0.80, class_id=0),
        ]
    }

    prec, rec, f1, ap = evaluate_single_iou(preds, gts, iou_threshold=0.5)
    assert rec == pytest.approx(1.0)
    assert prec == pytest.approx(0.5)
    assert f1 == pytest.approx(2 * 0.5 * 1.0 / (0.5 + 1.0))
    assert ap > 0.0


def test_evaluate_single_iou_greedy_matching_fallback() -> None:
    """Verifies greedy COCO matching: when GT0 is taken by pred0, pred1 matches GT1."""
    gts = {
        "img1": [
            Detection(x1=0.1, y1=0.1, x2=0.3, y2=0.3, score=1.0, class_id=0),  # GT 0
            Detection(x1=0.2, y1=0.1, x2=0.4, y2=0.3, score=1.0, class_id=0),  # GT 1
        ]
    }
    preds = {
        "img1": [
            # Pred 0 matches GT 0 with high IoU
            Detection(x1=0.1, y1=0.1, x2=0.3, y2=0.3, score=0.95, class_id=0),
            # Pred 1 overlaps GT 0 (IoU ~0.5) but also overlaps GT 1 (IoU ~0.8)
            Detection(x1=0.18, y1=0.1, x2=0.38, y2=0.3, score=0.85, class_id=0),
        ]
    }

    prec, rec, f1, ap = evaluate_single_iou(preds, gts, iou_threshold=0.5)
    # Both predictions must match their respective ground truths as TP
    assert prec == pytest.approx(1.0)
    assert rec == pytest.approx(1.0)
    assert f1 == pytest.approx(1.0)
    assert ap == pytest.approx(1.0)


def test_evaluate_detector_full() -> None:
    gts = {
        "img1": [Detection(x1=0.1, y1=0.1, x2=0.3, y2=0.3, score=1.0, class_id=0)],
        "img2": [Detection(x1=0.4, y1=0.4, x2=0.6, y2=0.6, score=1.0, class_id=0)],
    }
    preds = {
        "img1": [Detection(x1=0.1, y1=0.1, x2=0.3, y2=0.3, score=0.9, class_id=0)],
        "img2": [Detection(x1=0.4, y1=0.4, x2=0.6, y2=0.6, score=0.85, class_id=0)],
    }

    metrics = evaluate_detector(
        model_name="test_model",
        target_class="face",
        domain="overall",
        predictions_by_img=preds,
        ground_truths_by_img=gts,
        avg_latency_ms=12.5,
        params_millions=2.6,
        weight_size_mb=6.0,
    )

    assert metrics.map_50 == pytest.approx(1.0)
    assert metrics.precision_50 == pytest.approx(1.0)
    assert metrics.recall_50 == pytest.approx(1.0)
    assert metrics.avg_latency_ms == pytest.approx(12.5)
    assert metrics.throughput_fps == pytest.approx(80.0)


def test_benchmark_catalog_integrity() -> None:
    assert "fuyucchi_face" in BENCHMARK_CATALOG
    assert "zymk9_face" in BENCHMARK_CATALOG
    assert "deepghs_face_nano" in BENCHMARK_CATALOG
    assert "deepghs_head_nano" in BENCHMARK_CATALOG

    for name, info in BENCHMARK_CATALOG.items():
        assert "url" in info and info["url"].startswith("http")
        assert "filename" in info and info["filename"].endswith(".pt")
        assert "arch" in info
        assert "target" in info and info["target"] in ("face", "head")
        assert "type" in info and info["type"] in ("ultralytics", "yolov5")


def test_catnip_class_mapping_constants() -> None:
    assert CatnipDetector.CATNIP_CLASSES["body"] == 0
    assert CatnipDetector.CATNIP_CLASSES["head"] == 1
    assert CatnipDetector.CATNIP_CLASSES["face"] == 2


def test_get_detector_target_mismatch() -> None:
    """Verifies that requesting a head model for a face benchmark raises ValueError."""
    with pytest.raises(ValueError, match="cannot be used for benchmark target 'face'"):
        get_detector("deepghs_head_nano", target="face")

    with pytest.raises(ValueError, match="cannot be used for benchmark target 'head'"):
        get_detector("fuyucchi_face", target="head")


def test_classify_domain() -> None:
    assert classify_domain(Path("/data/manga109/slice_001.jpg")) == "manga_lineart"
    assert classify_domain(Path("/data/izutsumi/slice_002.png")) == "manga_lineart"
    assert classify_domain(Path("/data/ah_yolo/slice_003.jpg")) == "anime_illustration"
    assert classify_domain(Path("/data/anime_heads/slice_004.jpg")) == "anime_illustration"
    assert classify_domain(Path("/data/other/slice_005.jpg")) == "general"


def test_parse_yolo_label(tmp_path: Path) -> None:
    label_file = tmp_path / "sample.txt"
    # Format: class_id xc yc w h
    # Include valid target (class 2), non-target (class 0), and malformed line
    label_file.write_text("2 0.5 0.5 0.2 0.4\n0 0.1 0.1 0.1 0.1\nmalformed line\n")

    boxes = parse_yolo_label(label_file, target_gt_class=2)
    assert len(boxes) == 1
    b = boxes[0]
    assert b.x1 == pytest.approx(0.4)
    assert b.y1 == pytest.approx(0.3)
    assert b.x2 == pytest.approx(0.6)
    assert b.y2 == pytest.approx(0.7)
    assert b.score == 1.0
