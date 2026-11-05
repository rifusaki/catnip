"""Evaluation metrics engine for object detection (mAP50, mAP50-95, Precision, Recall)."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from src.eval.adapters import Detection

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """Metrics container for a detector's performance."""
    model_name: str
    target_class: str
    domain: str
    precision_50: float
    recall_50: float
    f1_50: float
    map_50: float
    map_75: float
    map_50_95: float
    total_images: int
    total_ground_truths: int
    total_predictions: int
    avg_latency_ms: float = 0.0
    throughput_fps: float = 0.0
    params_millions: float = 0.0
    weight_size_mb: float = 0.0

    def to_dict(self) -> dict[str, float | str | int]:
        return asdict(self)

    def to_markdown_row(self) -> str:
        return (
            f"| **{self.model_name}** | {self.params_millions:.1f}M | {self.weight_size_mb:.1f} MB | "
            f"{self.map_50:.4f} | {self.map_50_95:.4f} | {self.precision_50:.4f} | "
            f"{self.recall_50:.4f} | {self.f1_50:.4f} | {self.avg_latency_ms:.1f} ms |"
        )


# ============================================================================
# IoU Matrix Computation (Law 3 — Atomic Predictability)
# ============================================================================

def compute_iou(box1: Detection, box2: Detection) -> float:
    """Compute Intersection-over-Union (IoU) between two normalized boxes."""
    inter_x1 = max(box1.x1, box2.x1)
    inter_y1 = max(box1.y1, box2.y1)
    inter_x2 = min(box1.x2, box2.x2)
    inter_y2 = min(box1.y2, box2.y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    if inter_area <= 0.0:
        return 0.0

    area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
    area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area


def compute_iou_matrix(preds: Sequence[Detection], gts: Sequence[Detection]) -> np.ndarray:
    """Compute pairwise IoU matrix of shape [len(preds), len(gts)]."""
    n_preds = len(preds)
    n_gts = len(gts)
    if n_preds == 0 or n_gts == 0:
        return np.zeros((n_preds, n_gts), dtype=np.float32)

    matrix = np.zeros((n_preds, n_gts), dtype=np.float32)
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            matrix[i, j] = compute_iou(p, g)
    return matrix


# ============================================================================
# Average Precision & PR Curves (Law 1 — Early Exit, Law 3 — Predictability)
# ============================================================================

def compute_ap_from_pr(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute Average Precision using continuous all-points envelope interpolation."""
    if len(recalls) == 0 or len(precisions) == 0:
        return 0.0

    # Append sentinel bounding values [0, 1]
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute maximum precision envelope backwards
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Find points where recall changes
    indices = np.where(mrec[1:] != mrec[:-1])[0]

    # Sum area under envelope
    return float(np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1]))


def evaluate_single_iou(
    predictions_by_img: dict[str, list[Detection]],
    ground_truths_by_img: dict[str, list[Detection]],
    iou_threshold: float,
) -> tuple[float, float, float, float]:
    """Calculate (Precision, Recall, F1, AP) at a specific IoU threshold."""
    total_gts = sum(len(boxes) for boxes in ground_truths_by_img.values())

    flat_preds: list[tuple[str, Detection]] = []
    for img_id, pred_boxes in predictions_by_img.items():
        for p in pred_boxes:
            flat_preds.append((img_id, p))

    if total_gts == 0 or not flat_preds:
        return 0.0, 0.0, 0.0, 0.0

    # Sort descending by confidence score
    flat_preds.sort(key=lambda item: item[1].score, reverse=True)

    # Track matched ground truths per image to prevent double-matching
    matched_gt_indices: dict[str, set[int]] = {img_id: set() for img_id in ground_truths_by_img}

    tp = np.zeros(len(flat_preds), dtype=np.float32)
    fp = np.zeros(len(flat_preds), dtype=np.float32)

    for i, (img_id, pred) in enumerate(flat_preds):
        gts = ground_truths_by_img.get(img_id, [])
        if not gts:
            fp[i] = 1.0
            continue

        best_iou = 0.0
        best_gt_idx = -1

        # Greedy match strictly against currently unmatched ground truth boxes (COCO standard)
        for gt_idx, gt in enumerate(gts):
            if gt_idx in matched_gt_indices[img_id]:
                continue
            iou = compute_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp[i] = 1.0
            matched_gt_indices[img_id].add(best_gt_idx)
        else:
            fp[i] = 1.0

    # Cumulative sums
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    recalls = cum_tp / total_gts
    precisions = cum_tp / (cum_tp + cum_fp)

    ap = compute_ap_from_pr(recalls, precisions)
    final_prec = float(precisions[-1]) if len(precisions) > 0 else 0.0
    final_rec = float(recalls[-1]) if len(recalls) > 0 else 0.0
    f1 = (2 * final_prec * final_rec / (final_prec + final_rec)) if (final_prec + final_rec) > 0 else 0.0

    return final_prec, final_rec, f1, ap


# ============================================================================
# Full COCO-Style Multi-Threshold Evaluation
# ============================================================================

def evaluate_detector(
    model_name: str,
    target_class: str,
    domain: str,
    predictions_by_img: dict[str, list[Detection]],
    ground_truths_by_img: dict[str, list[Detection]],
    avg_latency_ms: float = 0.0,
    params_millions: float = 0.0,
    weight_size_mb: float = 0.0,
) -> EvaluationMetrics:
    """Run full evaluation across standard COCO IoU thresholds (0.50 to 0.95 in steps of 0.05)."""
    total_imgs = len(ground_truths_by_img)
    total_gts = sum(len(boxes) for boxes in ground_truths_by_img.values())
    total_preds = sum(len(boxes) for boxes in predictions_by_img.values())

    iou_levels = np.linspace(0.50, 0.95, 10)
    aps: list[float] = []
    prec_50, rec_50, f1_50, map_50 = 0.0, 0.0, 0.0, 0.0
    map_75 = 0.0

    for iou_thresh in iou_levels:
        p, r, f1, ap = evaluate_single_iou(predictions_by_img, ground_truths_by_img, float(iou_thresh))
        aps.append(ap)
        if abs(iou_thresh - 0.50) < 1e-4:
            prec_50, rec_50, f1_50, map_50 = p, r, f1, ap
        elif abs(iou_thresh - 0.75) < 1e-4:
            map_75 = ap

    map_50_95 = float(np.mean(aps)) if aps else 0.0
    fps = (1000.0 / avg_latency_ms) if avg_latency_ms > 0 else 0.0

    return EvaluationMetrics(
        model_name=model_name,
        target_class=target_class,
        domain=domain,
        precision_50=prec_50,
        recall_50=rec_50,
        f1_50=f1_50,
        map_50=map_50,
        map_75=map_75,
        map_50_95=map_50_95,
        total_images=total_imgs,
        total_ground_truths=total_gts,
        total_predictions=total_preds,
        avg_latency_ms=avg_latency_ms,
        throughput_fps=fps,
        params_millions=params_millions,
        weight_size_mb=weight_size_mb,
    )
