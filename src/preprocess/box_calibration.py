"""4-parameter face box convention adapter (loose external -> tight Manga109 faces).

External Danbooru-style face detectors (``deepghs_face_s``, ``zymk9_face``)
localize faces including the forehead and bangs, while ``Manga109`` and
``izutsumi`` annotate tight brow-to-chin ``<face>`` boxes. Before external
predictions can act as Stage 1 training labels, they must be mapped onto the
tight convention by the linear adapter

    x_hat = xc + alpha_x * w
    y_hat = yc + alpha_y * h
    w_hat = beta_w * w
    h_hat = beta_h * h

fitted on ``(pred_face, gt_face)`` pairs via Huber regression, which is robust
to loose-box outliers. A fitted adapter that would *degenerate* training labels
(scaling a box to a sliver or a balloon, or drifting it half a width away)
raises immediately instead of ever reaching a label file.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Calibration constants (Law 5 — Intentional Naming)
# ---------------------------------------------------------------------------

#: Minimum matched (pred, gt) pairs required to fit a trustworthy adapter.
MIN_MATCHED_PAIRS = 2

#: Hard sanity bounds on the fitted parameters. A degenerate adapter must
#: never become a training label, so both fit and load enforce these.
ALPHA_ABS_MAX = 0.5
BETA_MIN = 0.3
BETA_MAX = 3.0

#: Huber loss cut-off in units of the MAD scale (1.345 = 95% Gaussian
#: efficiency, the textbook default).
HUBER_DELTA_SCALE = 1.345

#: Factor converting a median absolute deviation to a standard deviation.
MAD_TO_STD = 1.4826


class FaceBoxAdapterError(ValueError):
    """Raised when an adapter is degenerate, unfittable, or unparseable."""


# ========================================================================
# Adapter  (Law 2 — Illegal adapters are unrepresentable)
# ========================================================================

@dataclass(frozen=True)
class FaceBoxAdapter:
    """Guarded linear adapter between loose and tight face box conventions.

    The constructor validates the parameters, so an instance can never
    represent a degenerate transform: centered shifts stay within half a
    box and scale factors stay inside ``[BETA_MIN, BETA_MAX]``.
    """

    alpha_x: float
    alpha_y: float
    beta_w: float
    beta_h: float

    def __post_init__(self) -> None:
        params = dict(alpha_x=self.alpha_x, alpha_y=self.alpha_y, beta_w=self.beta_w, beta_h=self.beta_h)
        for name, value in params.items():
            if not np.isfinite(value):
                raise FaceBoxAdapterError(f"FaceBoxAdapter parameter '{name}' must be finite, got {value!r}")

        for axis_name in ("alpha_x", "alpha_y"):
            if abs(params[axis_name]) > ALPHA_ABS_MAX:
                raise FaceBoxAdapterError(
                    f"Degenerate adapter: |{axis_name}|={abs(params[axis_name]):.4f} "
                    f"exceeds the ±{ALPHA_ABS_MAX} center-shift bound"
                )

        for scale_name in ("beta_w", "beta_h"):
            if not BETA_MIN <= params[scale_name] <= BETA_MAX:
                raise FaceBoxAdapterError(
                    f"Degenerate adapter: {scale_name}={params[scale_name]:.4f} "
                    f"outside the sane scaling range [{BETA_MIN}, {BETA_MAX}]"
                )

    def __repr__(self) -> str:
        return (
            f"FaceBoxAdapter(alpha_x={self.alpha_x:+.4f}, alpha_y={self.alpha_y:+.4f}, "
            f"beta_w={self.beta_w:.4f}, beta_h={self.beta_h:.4f})"
        )

    # ------------------------------------------------------------------
    # Transform  (Law 3 — pure: same input, same output)
    # ------------------------------------------------------------------

    def transform(self, box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """Map a normalized ``(xc, yc, w, h)`` box onto the tight convention."""
        xc, yc, w, h = box
        return (xc + self.alpha_x * w, yc + self.alpha_y * h, self.beta_w * w, self.beta_h * h)

    def transform_batch(self, boxes: np.ndarray) -> np.ndarray:
        """Vectorized :meth:`transform` over a validated ``(N, 4)`` array."""
        checked = parse_box_array(boxes, expected="boxes to transform")
        xc, yc, w, h = checked[:, 0], checked[:, 1], checked[:, 2], checked[:, 3]
        return np.stack(
            [xc + self.alpha_x * w, yc + self.alpha_y * h, self.beta_w * w, self.beta_h * h],
            axis=1,
        )

    # ------------------------------------------------------------------
    # Serialization  (Law 4 — refuse to load degenerate adapters)
    # ------------------------------------------------------------------

    def to_json(self, path: str | Path) -> None:
        """Write the adapter parameters to *path* as JSON (parents created)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "alpha_x": self.alpha_x,
            "alpha_y": self.alpha_y,
            "beta_w": self.beta_w,
            "beta_h": self.beta_h,
        }
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
        logger.info("Wrote face box adapter -> %s", target)

    @classmethod
    def from_json(cls, path: str | Path) -> FaceBoxAdapter:
        """Load an adapter from JSON.

        Raises:
            FaceBoxAdapterError: if the file is unreadable, malformed, or
                contains a degenerate (out-of-bounds) parameter set.
        """
        target = Path(path)
        try:
            raw = target.read_text(encoding="utf-8")
        except OSError as err:
            raise FaceBoxAdapterError(f"Cannot read face box adapter JSON '{target}': {err}") from err

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as err:
            raise FaceBoxAdapterError(f"Malformed face box adapter JSON '{target}': {err}") from err

        if not isinstance(payload, dict):
            raise FaceBoxAdapterError(f"Face box adapter JSON '{target}' must be a JSON object")

        required = ("alpha_x", "alpha_y", "beta_w", "beta_h")
        missing = [name for name in required if name not in payload]
        if missing:
            raise FaceBoxAdapterError(f"Face box adapter JSON '{target}' is missing fields: {missing}")

        try:
            params = {name: float(payload[name]) for name in required}
        except (TypeError, ValueError) as err:
            raise FaceBoxAdapterError(
                f"Face box adapter JSON '{target}' has non-numeric parameters: {err}"
            ) from err

        return cls(**params)


# ========================================================================
# Input parsing  (Law 2 — parse at the boundary, before any math)
# ========================================================================

def parse_box_array(boxes: np.ndarray, expected: str = "boxes") -> np.ndarray:
    """Validate and return an ``(N, 4)`` float array of ``(xc, yc, w, h)`` boxes."""
    arr = np.asarray(boxes, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise FaceBoxAdapterError(
            f"Expected {expected} as an (N, 4) array of (xc, yc, w, h), got shape {arr.shape}"
        )
    if arr.shape[0] == 0:
        raise FaceBoxAdapterError(f"Expected at least one row in {expected}, got an empty array")
    if not np.isfinite(arr).all():
        raise FaceBoxAdapterError(f"{expected} contain non-finite values")
    if (arr[:, 2] <= 0).any() or (arr[:, 3] <= 0).any():
        raise FaceBoxAdapterError(f"{expected} contain non-positive box widths or heights")
    return arr


def _cxcywh_iou_matrix(predictions: np.ndarray, ground_truths: np.ndarray) -> np.ndarray:
    """Pairwise IoU between ``(N, 4)`` and ``(M, 4)`` center-format arrays."""
    pred_x1 = predictions[:, 0] - predictions[:, 2] / 2.0
    pred_y1 = predictions[:, 1] - predictions[:, 3] / 2.0
    pred_x2 = predictions[:, 0] + predictions[:, 2] / 2.0
    pred_y2 = predictions[:, 1] + predictions[:, 3] / 2.0

    gt_x1 = ground_truths[:, 0] - ground_truths[:, 2] / 2.0
    gt_y1 = ground_truths[:, 1] - ground_truths[:, 3] / 2.0
    gt_x2 = ground_truths[:, 0] + ground_truths[:, 2] / 2.0
    gt_y2 = ground_truths[:, 1] + ground_truths[:, 3] / 2.0

    inter_w = np.maximum(
        0.0, np.minimum(pred_x2[:, None], gt_x2[None, :]) - np.maximum(pred_x1[:, None], gt_x1[None, :])
    )
    inter_h = np.maximum(
        0.0, np.minimum(pred_y2[:, None], gt_y2[None, :]) - np.maximum(pred_y1[:, None], gt_y1[None, :])
    )
    inter = inter_w * inter_h

    pred_area = (pred_x2 - pred_x1)[:, None] * (pred_y2 - pred_y1)[:, None]
    gt_area = (gt_x2 - gt_x1)[None, :] * (gt_y2 - gt_y1)[None, :]
    union = pred_area + gt_area - inter

    return np.where(union > 0, inter / union, 0.0)


def pair_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """IoU between two single ``(xc, yc, w, h)`` boxes."""
    matrix = _cxcywh_iou_matrix(
        np.asarray([a], dtype=float), np.asarray([b], dtype=float)
    )
    return float(matrix[0, 0])


def match_pairs(
    predictions: np.ndarray,
    ground_truths: np.ndarray,
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    """Greedy one-to-one matching by IoU.

    Returns ``(pred_idx, gt_idx, iou)`` triples, best IoU first. Ties break on
    the lower index pair so the result is fully deterministic.
    """
    iou = _cxcywh_iou_matrix(predictions, ground_truths)
    candidates = [
        (int(pi), int(gi), float(iou[pi, gi]))
        for pi in range(iou.shape[0])
        for gi in range(iou.shape[1])
        if iou[pi, gi] >= iou_threshold
    ]
    candidates.sort(key=lambda t: (-t[2], t[0], t[1]))

    taken_pred: set[int] = set()
    taken_gt: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for pi, gi, overlap in candidates:
        if pi in taken_pred or gi in taken_gt:
            continue
        taken_pred.add(pi)
        taken_gt.add(gi)
        matches.append((pi, gi, overlap))

    return matches


# ========================================================================
# Huber regression  (Law 3 — pure, no sklearn dependency)
# ========================================================================

def _huber_location(values: np.ndarray, max_iterations: int = 50, tolerance: float = 1e-10) -> float:
    """Huber M-estimator of a constant model: robust location of *values*.

    Minimizes ``sum_i huber(x_i - theta)`` by iteratively reweighted least
    squares, with the cut-off scaled by the median absolute deviation. For a
    zero-MAD sample (all values identical) the median is returned directly.
    """
    center = float(np.median(values))
    deviations = np.abs(values - np.median(values))
    scale = MAD_TO_STD * float(np.median(deviations))
    if scale <= 1e-12:
        return center

    delta = HUBER_DELTA_SCALE * scale
    for _ in range(max_iterations):
        residuals = np.abs(values - center)
        weights = np.where(residuals <= delta, 1.0, delta / np.maximum(residuals, 1e-12))
        next_center = float(np.sum(weights * values) / np.sum(weights))
        if abs(next_center - center) < tolerance:
            return next_center
        center = next_center

    return center


# ========================================================================
# Fitting & validation  (Law 4 — fail loud on degenerate output)
# ========================================================================

def fit_face_box_adapter(
    predictions: np.ndarray,
    ground_truths: np.ndarray,
    match_iou: float = 0.45,
) -> FaceBoxAdapter:
    """Fit the 4-parameter adapter from matched (prediction, ground truth) pairs.

    Args:
        predictions: ``(N, 4)`` normalized loose-face boxes ``(xc, yc, w, h)``.
        ground_truths: ``(M, 4)`` normalized tight-face boxes.
        match_iou: minimum IoU for a greedy one-to-one match.

    Returns:
        FaceBoxAdapter: parameters fitted by Huber regression.

    Raises:
        FaceBoxAdapterError: if fewer than ``MIN_MATCHED_PAIRS`` pairs match,
            or the fitted parameters leave the sane bounds.

    WARNING: predictions and ground_truths must come from ONE image (or a
    corpus where cross-matching is impossible). Greedy matching operates on
    flattened pools, so concatenated multi-image pools can pair a prediction
    with a ground truth from a DIFFERENT image. For already-paired boxes
    across many images use :func:`fit_face_box_adapter_from_pairs`.
    """
    preds = parse_box_array(predictions, expected="predictions")
    gts = parse_box_array(ground_truths, expected="ground_truths")

    pairs = match_pairs(preds, gts, iou_threshold=match_iou)
    if len(pairs) < MIN_MATCHED_PAIRS:
        raise FaceBoxAdapterError(
            f"Face box adapter needs >= {MIN_MATCHED_PAIRS} matched pairs at "
            f"IoU >= {match_iou}, matched only {len(pairs)} "
            f"({len(preds)} predictions vs {len(gts)} ground truths)"
        )

    aligned_preds = np.asarray([preds[pi] for pi, _, _ in pairs], dtype=np.float64)
    aligned_gts = np.asarray([gts[gi] for _, gi, _ in pairs], dtype=np.float64)
    return fit_face_box_adapter_from_pairs(aligned_preds, aligned_gts)


def _regression_target_ratios(pair: np.ndarray, ground_truth: np.ndarray) -> tuple[float, float, float, float]:
    """Per-pair (alpha_x, alpha_y, beta_w, beta_h) that map pair -> ground truth."""
    return (
        float((ground_truth[0] - pair[0]) / pair[2]),
        float((ground_truth[1] - pair[1]) / pair[3]),
        float(ground_truth[2] / pair[2]),
        float(ground_truth[3] / pair[3]),
    )


def fit_face_box_adapter_from_pairs(
    predictions: np.ndarray,
    ground_truths: np.ndarray,
) -> FaceBoxAdapter:
    """Fit the adapter from ALREADY-PAIRED boxes (row i <-> row i).

    Unlike :func:`fit_face_box_adapter`, no IoU matching happens here: row
    ``i`` of *predictions* must already correspond to row ``i`` of
    *ground_truths* (e.g. pairs matched per image upstream). This is the
    correct entry point for a multi-image corpus — greedy pool matching on
    concatenated boxes would pair predictions with ground truths from other
    images, scrambling the fit.

    Raises:
        FaceBoxAdapterError: on row-count mismatch, insufficient pairs, or a
            degenerate fit.
    """
    preds = parse_box_array(predictions, expected="predictions")
    gts = parse_box_array(ground_truths, expected="ground_truths")
    if len(preds) != len(gts):
        raise FaceBoxAdapterError(
            f"Aligned pair fit needs equal-length arrays, got {len(preds)} "
            f"predictions vs {len(gts)} ground truths"
        )
    if len(preds) < MIN_MATCHED_PAIRS:
        raise FaceBoxAdapterError(
            f"Face box adapter needs >= {MIN_MATCHED_PAIRS} aligned pairs, "
            f"got {len(preds)}"
        )

    targets = np.asarray(
        [_regression_target_ratios(p, g) for p, g in zip(preds, gts)],
        dtype=np.float64,
    )
    adapter = FaceBoxAdapter(
        alpha_x=_huber_location(targets[:, 0]),
        alpha_y=_huber_location(targets[:, 1]),
        beta_w=_huber_location(targets[:, 2]),
        beta_h=_huber_location(targets[:, 3]),
    )  # the constructor raises if the fit is degenerate
    logger.info("Fitted %s from %d aligned pairs", adapter, len(preds))
    return adapter


def validate_adapter(
    adapter: FaceBoxAdapter,
    predictions: np.ndarray,
    ground_truths: np.ndarray,
    match_iou: float = 0.45,
    iou_threshold: float = 0.75,
    min_quality: float = 0.45,
) -> dict:
    """Measure adapter quality; refuse (raise) below the *min_quality* floor.

    Metrics:
        matched: greedy one-to-one pairs at ``match_iou``.
        mean_iou_before / mean_iou_after: mean IoU of the matched pairs,
            computed against raw vs adapter-transformed predictions.
        recall_at_iou_before / recall_at_iou_after: fraction of ground truths
            matched at ``iou_threshold`` by raw vs transformed predictions.

    Raises:
        FaceBoxAdapterError: on empty inputs, zero matches, or when
            ``recall_at_iou_after < min_quality`` (a degenerate adapter must
            never become training labels).
    """
    preds = parse_box_array(predictions, expected="predictions")
    gts = parse_box_array(ground_truths, expected="ground_truths")

    pairs = match_pairs(preds, gts, iou_threshold=match_iou)
    if not pairs:
        raise FaceBoxAdapterError(
            f"Cannot validate adapter: 0 of {len(gts)} ground truths matched at IoU >= {match_iou}"
        )

    mean_iou_before = float(np.mean([overlap for _, _, overlap in pairs]))
    transformed_ious = [
        pair_iou(adapter.transform(tuple(preds[pi])), tuple(gts[gi]))
        for pi, gi, _ in pairs
    ]
    mean_iou_after = float(np.mean(transformed_ious))

    recall_before = _fraction_matched_at(preds, gts, iou_threshold)
    transformed_preds = adapter.transform_batch(preds)
    recall_after = _fraction_matched_at(transformed_preds, gts, iou_threshold)

    if recall_after < min_quality:
        raise FaceBoxAdapterError(
            f"Adapter validation failed the quality floor: recall_at_iou_after="
            f"{recall_after:.4f} < min_quality={min_quality:.4f} at IoU >= {iou_threshold} "
            f"(matched={len(pairs)}, mean_iou {mean_iou_before:.4f} -> {mean_iou_after:.4f}). "
            f"Refusing a degenerate adapter: {adapter}"
        )

    return {
        "matched": len(pairs),
        "mean_iou_before": mean_iou_before,
        "mean_iou_after": mean_iou_after,
        "recall_at_iou_before": recall_before,
        "recall_at_iou_after": recall_after,
    }


def validate_adapter_from_pairs(
    adapter: FaceBoxAdapter,
    predictions: np.ndarray,
    ground_truths: np.ndarray,
    iou_threshold: float = 0.75,
    min_quality: float = 0.45,
) -> dict:
    """Validate on ALIGNED pairs (row i <-> row i), without re-matching.

    Multi-image corpora must validate per-pair: greedy pool matching over
    concatenated boxes can pair a transformed prediction with a ground truth
    from another image, inflating ``recall_after`` and letting a bad adapter
    through the quality floor. Here recall is the fraction of aligned pairs
    whose TRANSFORMED prediction reaches ``iou_threshold`` with its own
    ground truth.

    Raises:
        FaceBoxAdapterError: on empty inputs or below the *min_quality* floor.
    """
    preds = parse_box_array(predictions, expected="predictions")
    gts = parse_box_array(ground_truths, expected="ground_truths")
    if len(preds) != len(gts):
        raise FaceBoxAdapterError(
            f"Aligned pair validation needs equal-length arrays, got "
            f"{len(preds)} predictions vs {len(gts)} ground truths"
        )
    if len(preds) == 0:
        raise FaceBoxAdapterError("Cannot validate adapter on zero aligned pairs")

    ious_before = np.asarray(
        [pair_iou(tuple(p), tuple(g)) for p, g in zip(preds, gts)], dtype=np.float64
    )
    transformed = adapter.transform_batch(preds)
    ious_after = np.asarray(
        [pair_iou(tuple(t), tuple(g)) for t, g in zip(transformed, gts)],
        dtype=np.float64,
    )

    recall_before = float(np.mean(ious_before >= iou_threshold))
    recall_after = float(np.mean(ious_after >= iou_threshold))
    if recall_after < min_quality:
        raise FaceBoxAdapterError(
            f"Adapter validation failed the quality floor: aligned recall "
            f"{recall_after:.4f} < min_quality={min_quality:.4f} at IoU >= "
            f"{iou_threshold} (pairs={len(preds)}, mean IoU "
            f"{float(ious_before.mean()):.4f} -> {float(ious_after.mean()):.4f}). "
            f"Refusing a degenerate adapter: {adapter}"
        )

    return {
        "matched": len(preds),
        "pairs": len(preds),
        "mean_iou_before": float(ious_before.mean()),
        "mean_iou_after": float(ious_after.mean()),
        "recall_at_iou_before": recall_before,
        "recall_at_iou_after": recall_after,
    }


def _fraction_matched_at(predictions: np.ndarray, ground_truths: np.ndarray, iou_threshold: float) -> float:
    """Fraction of ground truths covered one-to-one at *iou_threshold*."""
    matches = match_pairs(predictions, ground_truths, iou_threshold=iou_threshold)
    return len(matches) / len(ground_truths)
