"""Synthetic tests for the 4-parameter face box adapter (src/preprocess/box_calibration.py).

All inputs are synthetic normalized boxes — no images, no models, no network.
"""

import json

import numpy as np
import pytest

from src.preprocess.box_calibration import (
    ALPHA_ABS_MAX,
    BETA_MAX,
    BETA_MIN,
    FaceBoxAdapter,
    FaceBoxAdapterError,
    _huber_location,
    fit_face_box_adapter,
    pair_iou,
    validate_adapter,
)


def sample_ground_truths(count: int, seed: int = 0) -> np.ndarray:
    """Draw *count* non-overlapping-ish tight face boxes in unit space."""
    rng = np.random.RandomState(seed)
    widths = rng.uniform(0.06, 0.18, count)
    heights = widths * rng.uniform(0.9, 1.4, count)
    xcs = rng.uniform(0.25, 0.75, count)
    ycs = rng.uniform(0.25, 0.75, count)
    return np.stack([xcs, ycs, widths, heights], axis=1)


def inverse_transform(gts: np.ndarray, adapter: FaceBoxAdapter, noise: np.ndarray | None = None) -> np.ndarray:
    """Produce the loose predictions that *adapter* would map onto *gts*."""
    preds = np.empty_like(gts)
    preds[:, 2] = gts[:, 2] / adapter.beta_w
    preds[:, 3] = gts[:, 3] / adapter.beta_h
    preds[:, 0] = gts[:, 0] - adapter.alpha_x * preds[:, 2]
    preds[:, 1] = gts[:, 1] - adapter.alpha_y * preds[:, 3]
    if noise is not None:
        preds = preds + noise
    return preds


# ========================================================================
# FaceBoxAdapter
# ========================================================================

def test_transform_applies_the_linear_model() -> None:
    adapter = FaceBoxAdapter(alpha_x=0.1, alpha_y=-0.2, beta_w=2.0, beta_h=0.5)
    transformed = adapter.transform((0.5, 0.5, 0.2, 0.3))
    pytest.approx(transformed, abs=1e-12)
    assert transformed == pytest.approx((0.52, 0.44, 0.40, 0.15))


def test_transform_is_pure() -> None:
    adapter = FaceBoxAdapter(alpha_x=0.05, alpha_y=0.05, beta_w=1.1, beta_h=1.2)
    box = (0.4, 0.4, 0.2, 0.2)
    first = adapter.transform(box)
    second = adapter.transform(box)
    assert first == second
    assert box == (0.4, 0.4, 0.2, 0.2)  # input untouched


def test_constructor_guards_degenerate_parameters() -> None:
    valid = FaceBoxAdapter(alpha_x=0.0, alpha_y=0.0, beta_w=1.0, beta_h=1.0)
    assert valid.beta_w == 1.0  # sanity: identity is legal

    with pytest.raises(ValueError, match="Degenerate adapter: \\|alpha_x\\|"):
        FaceBoxAdapter(alpha_x=ALPHA_ABS_MAX + 0.01, alpha_y=0.0, beta_w=1.0, beta_h=1.0)

    with pytest.raises(ValueError, match="alpha_y"):
        FaceBoxAdapter(alpha_x=0.0, alpha_y=0.9, beta_w=1.0, beta_h=1.0)

    with pytest.raises(ValueError, match="beta_w"):
        FaceBoxAdapter(alpha_x=0.0, alpha_y=0.0, beta_w=BETA_MIN - 0.01, beta_h=1.0)

    with pytest.raises(ValueError, match="beta_h.*outside"):
        FaceBoxAdapter(alpha_x=0.0, alpha_y=0.0, beta_w=1.0, beta_h=BETA_MAX + 0.01)

    with pytest.raises(ValueError, match="finite"):
        FaceBoxAdapter(alpha_x=float("nan"), alpha_y=0.0, beta_w=1.0, beta_h=1.0)


def test_constructor_accepts_boundaries() -> None:
    edge = FaceBoxAdapter(alpha_x=ALPHA_ABS_MAX, alpha_y=-ALPHA_ABS_MAX, beta_w=BETA_MIN, beta_h=BETA_MAX)
    assert edge.alpha_x == ALPHA_ABS_MAX
    assert edge.beta_h == BETA_MAX


# ========================================================================
# JSON roundtrip
# ========================================================================

def test_json_roundtrip(tmp_path) -> None:
    adapter = FaceBoxAdapter(alpha_x=0.02, alpha_y=-0.04, beta_w=0.82, beta_h=0.75)
    path = tmp_path / "face_box_adapter.json"
    adapter.to_json(path)

    loaded = FaceBoxAdapter.from_json(path)
    assert loaded == adapter

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {
        "alpha_x": 0.02,
        "alpha_y": -0.04,
        "beta_w": 0.82,
        "beta_h": 0.75,
    }


def test_from_json_rejects_bad_payloads(tmp_path) -> None:
    degenerate = tmp_path / "degenerate.json"
    degenerate.write_text(json.dumps({"alpha_x": 0.0, "alpha_y": 0.0, "beta_w": 0.1, "beta_h": 1.0}))
    with pytest.raises(ValueError, match="Degenerate adapter: beta_w"):
        FaceBoxAdapter.from_json(degenerate)

    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps({"alpha_x": 0.0, "beta_w": 1.0}))
    with pytest.raises(ValueError, match="missing fields"):
        FaceBoxAdapter.from_json(missing)

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not valid json")
    with pytest.raises(ValueError, match="Malformed"):
        FaceBoxAdapter.from_json(malformed)

    with pytest.raises(ValueError, match="Cannot read"):
        FaceBoxAdapter.from_json(tmp_path / "nope.json")


# ========================================================================
# Fitting
# ========================================================================

def test_fit_recovers_known_parameters_exactly() -> None:
    true_adapter = FaceBoxAdapter(alpha_x=0.02, alpha_y=-0.04, beta_w=0.82, beta_h=0.75)
    gts = sample_ground_truths(50, seed=1)
    preds = inverse_transform(gts, true_adapter)

    fitted = fit_face_box_adapter(preds, gts)

    assert fitted.alpha_x == pytest.approx(true_adapter.alpha_x, abs=1e-6)
    assert fitted.alpha_y == pytest.approx(true_adapter.alpha_y, abs=1e-6)
    assert fitted.beta_w == pytest.approx(true_adapter.beta_w, abs=1e-6)
    assert fitted.beta_h == pytest.approx(true_adapter.beta_h, abs=1e-6)


def test_fit_recovers_known_parameters_under_noise() -> None:
    true_adapter = FaceBoxAdapter(alpha_x=0.02, alpha_y=-0.04, beta_w=0.82, beta_h=0.75)
    gts = sample_ground_truths(200, seed=2)
    rng = np.random.RandomState(3)
    noise = np.zeros_like(gts)
    noise[:, 0] = rng.normal(0.0, 0.004, len(gts))  # center jitter
    noise[:, 1] = rng.normal(0.0, 0.004, len(gts))
    noise[:, 2] = rng.normal(0.0, 0.02 * gts[:, 2], len(gts))  # 2% width jitter
    noise[:, 3] = rng.normal(0.0, 0.02 * gts[:, 3], len(gts))
    preds = inverse_transform(gts, true_adapter, noise=noise)

    fitted = fit_face_box_adapter(preds, gts)

    assert fitted.alpha_x == pytest.approx(true_adapter.alpha_x, abs=0.05)
    assert fitted.alpha_y == pytest.approx(true_adapter.alpha_y, abs=0.05)
    assert fitted.beta_w == pytest.approx(true_adapter.beta_w, abs=0.05)
    assert fitted.beta_h == pytest.approx(true_adapter.beta_h, abs=0.05)


def test_fit_is_deterministic() -> None:
    true_adapter = FaceBoxAdapter(alpha_x=0.01, alpha_y=0.03, beta_w=0.9, beta_h=1.1)
    gts = sample_ground_truths(40, seed=4)
    preds = inverse_transform(gts, true_adapter)

    first = fit_face_box_adapter(preds, gts)
    second = fit_face_box_adapter(preds, gts)
    assert first == second


def test_fit_rejects_insufficient_matches() -> None:
    gts = sample_ground_truths(5, seed=5)

    # Single matched pair (identity prediction on one GT).
    single_pair = np.stack([gts[0], [9.0, 9.0, 0.1, 0.1]])  # second row cannot match
    with pytest.raises(ValueError, match="matched only 1"):
        fit_face_box_adapter(single_pair, gts)

    # No matches at all: predictions pushed far outside every ground truth.
    far_away = sample_ground_truths(5, seed=6)
    far_away[:, 0] += 10.0
    with pytest.raises(ValueError, match="matched only 0"):
        fit_face_box_adapter(far_away, gts)


def test_fit_rejects_degenerate_fitted_parameters() -> None:
    # Boxes scaled 8x larger than their matched GTs only match at a tiny IoU
    # threshold, exposing the degenerate-scale guard wiring in the fit path.
    gts = np.array(
        [[0.5, 0.5, 0.10, 0.10], [0.7, 0.5, 0.08, 0.12], [0.5, 0.7, 0.12, 0.08]]
    )
    preds = np.array(
        [[0.5, 0.5, 0.80, 0.80], [0.7, 0.5, 0.64, 0.96], [0.5, 0.7, 0.96, 0.64]]
    )
    with pytest.raises(ValueError, match="Degenerate adapter: beta_w"):
        fit_face_box_adapter(preds, gts, match_iou=0.01)


def test_fit_rejects_malformed_inputs() -> None:
    gts = sample_ground_truths(5, seed=7)
    with pytest.raises(ValueError, match=r"\(N, 4\)"):
        fit_face_box_adapter(np.zeros((5, 5)), gts)
    with pytest.raises(ValueError, match="empty"):
        fit_face_box_adapter(np.zeros((0, 4)), gts)
    with pytest.raises(ValueError, match="non-positive"):
        fit_face_box_adapter(np.array([[0.5, 0.5, 0.0, 0.1]]), gts)


def test_huber_location_is_robust_to_outliers() -> None:
    rng = np.random.RandomState(0)
    inliers = rng.normal(0.0, 0.02, 60)
    contaminated = np.concatenate([inliers, [1.5, 1.5, 1.0]])

    location = _huber_location(contaminated)
    plain_mean = float(np.mean(contaminated))

    assert abs(location) < 0.05  # stays on the inlier cluster
    assert abs(location) < abs(plain_mean)


def test_huber_location_returns_median_for_identical_values() -> None:
    assert _huber_location(np.full(10, 0.37)) == pytest.approx(0.37)


# ========================================================================
# Validation
# ========================================================================

def good_pairs() -> tuple[np.ndarray, np.ndarray, FaceBoxAdapter]:
    true_adapter = FaceBoxAdapter(alpha_x=0.02, alpha_y=-0.04, beta_w=0.82, beta_h=0.75)
    gts = sample_ground_truths(30, seed=8)
    rng = np.random.RandomState(9)
    noise = np.zeros_like(gts)
    noise[:, 0] = rng.normal(0.0, 0.002, len(gts))
    noise[:, 1] = rng.normal(0.0, 0.002, len(gts))
    noise[:, 2] = rng.normal(0.0, 0.01 * gts[:, 2], len(gts))
    noise[:, 3] = rng.normal(0.0, 0.01 * gts[:, 3], len(gts))
    return inverse_transform(gts, true_adapter, noise=noise), gts, true_adapter


def test_validate_adapter_passes_above_quality_floor() -> None:
    preds, gts, adapter = good_pairs()
    metrics = validate_adapter(adapter, preds, gts)

    assert metrics["matched"] == len(gts)
    assert metrics["recall_at_iou_after"] == pytest.approx(1.0)
    assert metrics["recall_at_iou_before"] == pytest.approx(0.0, abs=0.35)  # loose boxes rarely reach 0.75
    assert metrics["mean_iou_after"] > metrics["mean_iou_before"]


def test_validate_adapter_raises_below_quality_floor() -> None:
    # Identity adapter applied to half-shifted boxes: IoU ~0.5-0.6 at match
    # time but far below the 0.75 validation threshold before and after.
    gts = sample_ground_truths(20, seed=10)
    preds = gts.copy()
    preds[:, 0] += 0.35 * preds[:, 2]  # shift by ~35% of width

    identity = FaceBoxAdapter(alpha_x=0.0, alpha_y=0.0, beta_w=1.0, beta_h=1.0)
    with pytest.raises(ValueError, match="quality floor"):
        validate_adapter(identity, preds, gts)


def test_validate_adapter_raises_on_zero_matches() -> None:
    gts = sample_ground_truths(5, seed=11)
    far_away = gts.copy()
    far_away[:, 0] += 10.0
    identity = FaceBoxAdapter(alpha_x=0.0, alpha_y=0.0, beta_w=1.0, beta_h=1.0)

    with pytest.raises(ValueError, match="0 of 5 ground truths matched"):
        validate_adapter(identity, far_away, gts)


def test_pair_iou_examples() -> None:
    assert pair_iou((0.5, 0.5, 0.2, 0.2), (0.5, 0.5, 0.2, 0.2)) == pytest.approx(1.0)
    assert pair_iou((0.5, 0.5, 0.2, 0.2), (0.9, 0.5, 0.2, 0.2)) == pytest.approx(0.0)
    assert pair_iou((0.5, 0.5, 0.2, 0.2), (0.5, 0.5, 0.4, 0.4)) == pytest.approx(0.25)


def test_from_pairs_fit_is_not_scrambled_by_cross_image_matches() -> None:
    """Aligned-pair fit survives overlapping boxes from different images (C1).

    Image A's prediction loosely covers ITS ground truth (IoU 0.49, beta
    0.7×) while image B's prediction — which happens to sit at the exact
    coordinates of image A's ground truth — has IoU 1.0 with it. Greedy
    POOL matching therefore pairs B's prediction with A's ground truth and
    drops A's predictions entirely, fitting the identity instead of the
    per-image truth. The aligned fit must recover each pair's own targets.
    """
    from src.preprocess.box_calibration import fit_face_box_adapter_from_pairs

    pred_image_a = (0.50, 0.50, 0.20, 0.20)
    gt_image_a = (0.50, 0.50, 0.14, 0.14)    # beta 0.70, IoU(pred_a, gt_a) = 0.49
    pred_image_b = (0.50, 0.50, 0.14, 0.14)  # identical coords to gt_image_a
    gt_image_b = (0.50, 0.50, 0.10, 0.10)    # beta 5/7, own IoU 0.51 < 1.0

    preds = np.asarray(
        [pred_image_a, pred_image_a, pred_image_b, pred_image_b], dtype=np.float64
    )
    aligned_gts = np.asarray(
        [gt_image_a, gt_image_a, gt_image_b, gt_image_b], dtype=np.float64
    )

    aligned = fit_face_box_adapter_from_pairs(preds, aligned_gts)
    mixed_hubers = _huber_location(np.asarray([0.70, 0.70, 5.0 / 7.0, 5.0 / 7.0]))
    assert aligned.beta_w == pytest.approx(mixed_hubers)
    assert aligned.beta_h == pytest.approx(mixed_hubers)

    # Pool matching on the flattened arrays steals the 1.0-IoU cross-image
    # pairs and fits beta == 1.0 on them — the scrambled result.
    pool = fit_face_box_adapter(preds, aligned_gts, match_iou=0.45)
    assert pool.beta_w == pytest.approx(1.0)
    assert aligned.beta_w != pool.beta_w


def test_from_pairs_inputs_must_align_and_validate_without_rematching() -> None:
    from src.preprocess.box_calibration import (
        fit_face_box_adapter_from_pairs,
        validate_adapter_from_pairs,
    )

    preds = np.asarray([(0.49, 0.49, 0.14, 0.18)] * 2, dtype=np.float64)
    gts_wrong_length = np.asarray([(0.49, 0.49, 0.098, 0.126)], dtype=np.float64)
    with pytest.raises(ValueError, match="equal-length arrays"):
        fit_face_box_adapter_from_pairs(preds, gts_wrong_length)

    # Validated on its own pairs: a PERFECT adapter reaches recall 1.0, a
    # broken one is refused — no cross-pair rescue possible.
    gts = np.asarray([(0.49, 0.49, 0.098, 0.126)] * 2, dtype=np.float64)
    perfect = fit_face_box_adapter_from_pairs(preds, gts)
    report = validate_adapter_from_pairs(perfect, preds, gts, iou_threshold=0.75)
    assert report["recall_at_iou_after"] == pytest.approx(1.0)

    broken = FaceBoxAdapter(alpha_x=0.4, alpha_y=0.4, beta_w=1.4, beta_h=1.4)
    with pytest.raises(ValueError, match="quality floor"):
        validate_adapter_from_pairs(broken, preds, gts, iou_threshold=0.75)

    with pytest.raises(ValueError, match="equal-length arrays"):
        validate_adapter_from_pairs(perfect, preds, gts_wrong_length)
