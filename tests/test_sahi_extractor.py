"""Unit tests for the gated SAHI extractor (panel/page, WBF, hierarchy filter).

All tests are synthetic-data and CPU-only: the ``predictor`` seam replaces
model inference, so no YOLO weights, network access, or ``catnip-data``
content is required.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
from sahi.slicing import get_slice_bboxes

from src.config import SahiParams
from src.detection.sahi_extractor import (
    DetectedPart,
    ExtractionSummary,
    SahiExtractor,
    _RawDetection,
    apply_class_confidence_thresholds,
    clamp_box_to_image,
    compute_page_dimensions,
    compute_page_scale,
    decide_gate,
    filter_anatomical_hierarchy,
    find_image_files,
    fuse_detections_per_class,
    intersection_over_self,
    intersection_over_union,
    nmm_fusion,
    project_box_to_original_space,
    rescale_image,
    weighted_box_fusion,
)

# Parity with settings.params.sahi defaults in config/pipeline.yaml.
DEFAULT_PARAMS = dict(
    slice_height=640,
    slice_width=640,
    overlap_ratio=0.2,
    confidence_threshold=0.01,
    min_area_ratio=0.3,
    postprocess_type="WBF",
    postprocess_match_metric="IOS",
    postprocess_match_threshold=0.40,
    panel_max_dim=800,
    canonical_page_max_dim=1800,
    hierarchical_verify=True,
    class_conf_thresholds={"body": 0.52, "head": 0.52, "face": 0.56},
)


def make_params(**overrides) -> SahiParams:
    return SahiParams(**{**DEFAULT_PARAMS, **overrides})


def scripted_extractor(handler, **param_overrides):
    """Extractor whose inference is replaced by ``handler(shape, call_index)``.

    Returns ``(extractor, call_shapes)`` where ``call_shapes`` records the
    shape of every array passed to the predictor, in call order.
    """
    params = make_params(**param_overrides)
    call_shapes: list[tuple[int, int, int]] = []

    def predictor(image_rgb):
        call_shapes.append(tuple(image_rgb.shape))
        return handler(tuple(image_rgb.shape), len(call_shapes) - 1)

    extractor = SahiExtractor(
        model_path="unused.pt", sahi_params=params, predictor=predictor
    )
    return extractor, call_shapes


def part(x1, y1, x2, y2, score, class_name) -> DetectedPart:
    return DetectedPart(x1=x1, y1=y1, x2=x2, y2=y2, score=score, class_name=class_name)


def raw(x1, y1, x2, y2, score, class_id) -> _RawDetection:
    return _RawDetection(x1=x1, y1=y1, x2=x2, y2=y2, score=score, class_id=class_id)


def write_image(path: Path, height: int, width: int) -> None:
    """Write a synthetic solid image; content is irrelevant to extraction."""
    array = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(path), array)


def slice_windows_of(height, width, params: SahiParams) -> list[tuple[int, int, int, int]]:
    """Mirror of the extractor's slice grid (train/inference parity source)."""
    return [
        tuple(window)
        for window in get_slice_bboxes(
            image_height=height,
            image_width=width,
            slice_height=params.slice_height,
            slice_width=params.slice_width,
            overlap_height_ratio=params.overlap_ratio,
            overlap_width_ratio=params.overlap_ratio,
            auto_slice_resolution=False,
        )
    ]


# ---------------------------------------------------------------------------
# (a) Weighted Box Fusion
# ---------------------------------------------------------------------------


def test_wbf_fuses_overlapping_boxes_with_confidence_weighted_mean() -> None:
    boxes = [(0.0, 0.0, 10.0, 10.0), (2.0, 2.0, 12.0, 12.0)]
    scores = [0.9, 0.6]

    fused_boxes, fused_scores = weighted_box_fusion(boxes, scores, "IOS", 0.40)

    assert len(fused_boxes) == 1
    assert fused_scores[0] == pytest.approx(0.75)
    # Each coordinate is the score-weighted mean of the cluster members.
    assert fused_boxes[0][0] == pytest.approx((0.9 * 0.0 + 0.6 * 2.0) / 1.5)
    assert fused_boxes[0][1] == pytest.approx((0.9 * 0.0 + 0.6 * 2.0) / 1.5)
    assert fused_boxes[0][2] == pytest.approx((0.9 * 10.0 + 0.6 * 12.0) / 1.5)
    assert fused_boxes[0][3] == pytest.approx((0.9 * 10.0 + 0.6 * 12.0) / 1.5)


def test_wbf_keeps_disjoint_boxes() -> None:
    boxes = [(0.0, 0.0, 10.0, 10.0), (50.0, 50.0, 60.0, 60.0)]
    scores = [0.9, 0.6]

    fused_boxes, fused_scores = weighted_box_fusion(boxes, scores, "IOS", 0.40)

    assert fused_boxes == [(0.0, 0.0, 10.0, 10.0), (50.0, 50.0, 60.0, 60.0)]
    assert fused_scores == [0.9, 0.6]


def test_wbf_single_box_passes_through() -> None:
    fused_boxes, fused_scores = weighted_box_fusion(
        [(1.0, 2.0, 3.0, 4.0)], [0.5], "IOS", 0.40
    )
    assert fused_boxes == [(1.0, 2.0, 3.0, 4.0)]
    assert fused_scores == [0.5]


def test_wbf_empty_input_returns_empty_output() -> None:
    assert weighted_box_fusion([], [], "IOS", 0.40) == ([], [])


def test_wbf_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="equal length"):
        weighted_box_fusion([(0.0, 0.0, 1.0, 1.0)], [0.5, 0.4], "IOS", 0.40)


def test_wbf_unknown_match_metric_fails_loud() -> None:
    with pytest.raises(ValueError, match="Unknown match metric"):
        weighted_box_fusion([(0.0, 0.0, 1.0, 1.0)], [0.5], "DICE", 0.40)


def test_nmm_keeps_largest_box_per_cluster() -> None:
    boxes = [(0.0, 0.0, 10.0, 10.0), (2.0, 2.0, 12.0, 12.0), (50.0, 50.0, 60.0, 60.0)]
    scores = [0.9, 0.6, 0.4]

    kept_boxes, kept_scores = nmm_fusion(boxes, scores, "IOS", 0.40)

    # The overlapping pair collapses to its highest-scored member; the
    # disjoint box survives untouched.
    assert kept_boxes == [(0.0, 0.0, 10.0, 10.0), (50.0, 50.0, 60.0, 60.0)]
    assert kept_scores == [0.9, 0.4]


def test_nmm_empty_input_returns_empty_output() -> None:
    assert nmm_fusion([], [], "IOS", 0.40) == ([], [])


def test_nmm_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="equal length"):
        nmm_fusion([(0.0, 0.0, 1.0, 1.0)], [0.5, 0.4], "IOS", 0.40)


def test_fusion_never_merges_different_classes() -> None:
    detections = [
        raw(0, 0, 10, 10, 0.9, class_id=2),  # face
        raw(1, 1, 11, 11, 0.8, class_id=2),  # overlapping face -> fuses
        raw(0, 0, 10, 10, 0.7, class_id=1),  # head at the same spot -> stays
        raw(0, 0, 10, 10, 0.6, class_id=0),  # body at the same spot -> stays
    ]

    fused = fuse_detections_per_class(detections, "WBF", "IOS", 0.40)

    assert [d.class_id for d in fused] == [2, 1, 0]
    face = fused[0]
    assert face.score == pytest.approx(0.85)
    assert face.x1 == pytest.approx(0.8 / 1.7)  # (0.9*0 + 0.8*1) / 1.7
    assert face.x2 == pytest.approx((0.9 * 10 + 0.8 * 11) / 1.7)
    for other in fused[1:]:
        assert (other.x1, other.y1, other.x2, other.y2) == (0.0, 0.0, 10.0, 10.0)


def test_fusion_rejects_unknown_postprocess_type() -> None:
    with pytest.raises(ValueError, match="Unknown postprocess_type"):
        fuse_detections_per_class([], "BOGUS", "IOS", 0.40)


# ---------------------------------------------------------------------------
# (b) Panel/page gating, canonical scaling, coordinate projection
# ---------------------------------------------------------------------------


def test_gate_decision_boundaries() -> None:
    assert decide_gate(800, 800, 800) == "panel"
    assert decide_gate(300, 800, 800) == "panel"
    assert decide_gate(801, 700, 800) == "page"
    assert decide_gate(700, 9000, 800) == "page"
    with pytest.raises(ValueError, match="panel_max_dim"):
        decide_gate(100, 100, 0)
    with pytest.raises(ValueError, match="must be positive"):
        decide_gate(0, 100, 800)


def test_page_scale_never_upscales() -> None:
    assert compute_page_scale(900, 1600, 1800) == 1.0
    assert compute_page_scale(1800, 1800, 1800) == 1.0  # boundary stays at 1.0
    assert compute_page_scale(2400, 1600, 1800) == pytest.approx(0.75)
    with pytest.raises(ValueError, match="canonical_page_max_dim"):
        compute_page_scale(1000, 1000, 0)


def test_page_dimensions_round_but_never_collapse() -> None:
    assert compute_page_dimensions(1800, 1309, 1.0) == (1800, 1309)
    assert compute_page_dimensions(1000, 1000, 0.001) == (1, 1)
    with pytest.raises(ValueError, match="scale"):
        compute_page_dimensions(100, 100, 1.5)


def test_projection_offset_and_scale_math() -> None:
    projected = project_box_to_original_space(
        (28.0, 42.0, 128.0, 142.0), 669.0, 1160.0, 0.75
    )
    assert projected == pytest.approx(
        (697 / 0.75, 1202 / 0.75, 797 / 0.75, 1302 / 0.75)
    )
    assert project_box_to_original_space((1, 2, 3, 4), 0.0, 0.0, 1.0) == (1.0, 2.0, 3.0, 4.0)
    with pytest.raises(ValueError, match="scale"):
        project_box_to_original_space((1, 2, 3, 4), 0, 0, 1.5)


def test_rescale_image_is_identity_at_scale_one() -> None:
    array = np.zeros((1800, 1200, 3), dtype=np.uint8)
    assert rescale_image(array, 1.0) is array
    downscaled = rescale_image(np.zeros((2400, 1600, 3), dtype=np.uint8), 0.75)
    assert downscaled.shape == (1800, 1200, 3)


def test_panel_route_runs_one_direct_prediction(tmp_path) -> None:
    image_path = tmp_path / "panel.png"
    write_image(image_path, height=700, width=700)

    extractor, call_shapes = scripted_extractor(
        lambda shape, index: [(10, 20, 300, 400, 0.9, 1)]
    )

    parts = extractor.extract_from_image(image_path)

    assert call_shapes == [(700, 700, 3)]  # exactly one call on the full image
    assert parts == [part(10, 20, 300, 400, 0.9, "head")]


def test_page_route_scales_canonically_and_projects_back(tmp_path) -> None:
    # 2400x1600 input -> canonical 1800x1200 page (scale 0.75).
    image_path = tmp_path / "page.png"
    write_image(image_path, height=2400, width=1600)
    page_shape = (1800, 1200, 3)

    def handler(shape, _index):
        if shape == page_shape:
            return [(200, 300, 600, 1200, 0.90, 2)]
        return []

    extractor, call_shapes = scripted_extractor(handler)
    windows = slice_windows_of(1800, 1200, extractor.params)

    parts = extractor.extract_from_image(image_path)

    # One call per slice window plus the full-page context pass.
    assert len(call_shapes) == len(windows) + 1
    assert call_shapes[-1] == page_shape
    for shape in call_shapes[:-1]:
        assert max(shape[0], shape[1]) <= 640  # windows only
    # Page-space box maps back to original pixels: coords / 0.75.
    assert parts == [part(266, 400, 800, 1600, 0.9, "face")]


def test_page_route_does_not_upscale_below_canonical(tmp_path) -> None:
    # 1600x1400 input: sliced (page gate) but rescaled by exactly 1.0.
    image_path = tmp_path / "page.png"
    write_image(image_path, height=1600, width=1400)
    page_shape = (1600, 1400, 3)

    def handler(shape, _index):
        if shape == page_shape:
            return [(700, 800, 1200, 1500, 0.90, 0)]
        return []

    extractor, call_shapes = scripted_extractor(handler)
    windows = slice_windows_of(1600, 1400, extractor.params)

    parts = extractor.extract_from_image(image_path)

    assert len(windows) > 1  # IS sliced
    assert len(call_shapes) == len(windows) + 1
    assert call_shapes[-1] == (1600, 1400, 3)  # untouched dimensions
    assert parts == [part(700, 800, 1200, 1500, 0.9, "body")]  # scale 1.0


# ---------------------------------------------------------------------------
# (c) Per-class confidence thresholds
# ---------------------------------------------------------------------------


def test_thresholds_filter_per_class_with_inclusive_boundary() -> None:
    detections = [
        part(0, 0, 10, 10, 0.51, "body"),   # below body threshold -> dropped
        part(0, 0, 10, 10, 0.52, "body"),   # exactly at threshold -> kept
        part(0, 0, 10, 10, 0.52, "head"),   # kept
        part(0, 0, 10, 10, 0.55, "face"),   # below face threshold -> dropped
        part(0, 0, 10, 10, 0.56, "face"),   # exactly at threshold -> kept
    ]
    thresholds = {"body": 0.52, "head": 0.52, "face": 0.56}

    kept = apply_class_confidence_thresholds(detections, thresholds)

    assert kept == [detections[1], detections[2], detections[4]]


def test_thresholds_fail_loud_on_unknown_class() -> None:
    with pytest.raises(ValueError, match="No confidence threshold configured for class 'head'"):
        apply_class_confidence_thresholds([part(0, 0, 1, 1, 0.9, "head")], {"body": 0.5})


def test_pipeline_drops_detections_below_class_threshold_with_faces_deferred(
    tmp_path,
) -> None:
    image_path = tmp_path / "panel.png"
    write_image(image_path, height=700, width=700)

    def handler(_shape, _index):
        return [
            (10, 10, 100, 100, 0.40, 0),   # body 0.40 < 0.52 -> dropped
            (200, 200, 300, 300, 0.60, 0),  # body 0.60 -> kept
            (400, 400, 500, 500, 0.559, 2),  # dangling face 0.559 -> dropped by hierarchy
            (600, 300, 690, 400, 0.80, 2),  # dangling face 0.80 -> kept (hi band)
        ]

    extractor, _ = scripted_extractor(handler)
    parts = extractor.extract_from_image(image_path)

    # Dangling faces never have a head/body parent: only the >= 0.70
    # close-up band keeps one. Face class thresholds are deferred to the
    # hierarchy rule when hierarchical_verify is on (spec §2).
    assert parts == [
        part(200, 200, 300, 300, 0.6, "body"),
        part(600, 300, 690, 400, 0.8, "face"),
    ]


def test_pipeline_hierarchy_face_band_between_lo_and_class_threshold(
    tmp_path,
) -> None:
    """Faces in [0.45, 0.56) survive via containment, per spec §2.

    The class threshold (0.56) must not consume the hierarchy band's lower
    stretch: a 0.50 confidence face tightly inside its head is genuine
    signal, reachable only when the hierarchy governs faces. Same-class
    overlapping boxes would fuse, so each band case gets its own image.
    """
    image_path = tmp_path / "panel.png"
    write_image(image_path, height=700, width=700)

    def handler(_shape, _index):
        return [
            # head kept at 0.9 -> parent for contained faces
            (100, 100, 300, 300, 0.90, 1),
            # contained face at 0.50 (< face class threshold 0.56,
            # inside the hierarchy band) -> must be KEPT
            (150, 120, 240, 200, 0.50, 2),
            # dangling face at 0.60 in [0.56, 0.70) -> dropped (no parent)
            (600, 300, 690, 400, 0.60, 2),
        ]

    extractor, _ = scripted_extractor(handler)
    parts = extractor.extract_from_image(image_path)

    assert parts == [
        part(100, 100, 300, 300, 0.9, "head"),
        part(150, 120, 240, 200, 0.5, "face"),
    ]

    low_image = tmp_path / "low_face.png"
    write_image(low_image, height=700, width=700)

    def low_handler(_shape, _index):
        return [
            (100, 100, 300, 300, 0.90, 1),
            # contained face at 0.44 (< lo_conf 0.45) -> dropped
            (150, 120, 240, 200, 0.44, 2),
        ]

    extractor, _ = scripted_extractor(low_handler)
    assert extractor.extract_from_image(low_image) == [
        part(100, 100, 300, 300, 0.9, "head"),
    ]


# ---------------------------------------------------------------------------
# (d) Hierarchical anatomy filter
# ---------------------------------------------------------------------------


def test_hierarchy_keeps_contained_mid_confidence_face() -> None:
    face = part(0, 0, 10, 10, 0.55, "face")     # 90% inside the head
    head = part(0, 1, 10, 11, 0.9, "head")

    assert intersection_over_self((0, 0, 10, 10), (0, 1, 10, 11)) == pytest.approx(0.9)
    assert filter_anatomical_hierarchy([face, head], enabled=True) == [face, head]


def test_hierarchy_drops_uncontained_mid_confidence_face() -> None:
    lone_face = part(50, 50, 60, 60, 0.55, "face")
    far_head = part(0, 0, 10, 10, 0.9, "head")

    kept = filter_anatomical_hierarchy([lone_face, far_head], enabled=True)

    assert kept == [far_head]


def test_hierarchy_keeps_high_confidence_face_without_parent() -> None:
    big_face = part(0, 0, 700, 700, 0.80, "face")  # extreme close-up panel

    assert filter_anatomical_hierarchy([big_face], enabled=True) == [big_face]


def test_hierarchy_drops_low_confidence_face_even_when_contained() -> None:
    weak_face = part(0, 0, 10, 10, 0.40, "face")
    parent_head = part(0, 1, 10, 11, 0.9, "head")

    kept = filter_anatomical_hierarchy([weak_face, parent_head], enabled=True)

    assert kept == [parent_head]


def test_hierarchy_confidence_boundaries() -> None:
    hi_face = part(0, 0, 10, 10, 0.70, "face")       # hi boundary: unconditional
    lo_face = part(0, 0, 10, 10, 0.45, "face")      # lo boundary, dangling
    sub_lo_face = part(0, 0, 10, 10, 0.449, "face")  # strictly below lo

    assert filter_anatomical_hierarchy([hi_face], enabled=True) == [hi_face]
    assert filter_anatomical_hierarchy([lo_face], enabled=True) == []
    assert filter_anatomical_hierarchy([sub_lo_face], enabled=True) == []


def test_hierarchy_disabled_returns_all_detections() -> None:
    dangling = part(0, 0, 10, 10, 0.55, "face")
    weak = part(0, 0, 10, 10, 0.20, "face")

    assert filter_anatomical_hierarchy([dangling, weak], enabled=False) == [dangling, weak]


def test_hierarchy_passes_through_non_face_classes() -> None:
    body = part(0, 0, 100, 400, 0.30, "body")
    head = part(10, 0, 60, 50, 0.10, "head")

    assert filter_anatomical_hierarchy([body, head], enabled=True) == [body, head]


def test_hierarchy_face_cannot_act_as_parent() -> None:
    mid_face = part(0, 0, 10, 10, 0.55, "face")
    hi_face = part(0, 0, 10, 10, 0.90, "face")  # overlaps but is a face, not a parent

    kept = filter_anatomical_hierarchy([mid_face, hi_face], enabled=True)

    assert kept == [hi_face]


def test_hierarchy_body_counts_as_parent() -> None:
    face = part(100, 100, 130, 160, 0.55, "face")
    body = part(0, 0, 400, 600, 0.8, "body")

    assert filter_anatomical_hierarchy([face, body], enabled=True) == [face, body]


def test_ios_and_iou_geometry() -> None:
    outer = (0.0, 0.0, 10.0, 10.0)
    inner = (2.0, 2.0, 8.0, 8.0)
    assert intersection_over_self(inner, outer) == pytest.approx(1.0)
    assert intersection_over_self(outer, inner) == pytest.approx(0.36)
    assert intersection_over_union(inner, outer) == pytest.approx(0.36)
    assert intersection_over_self((0.0, 0.0, 0.0, 5.0), outer) == 0.0  # degenerate


# ---------------------------------------------------------------------------
# (e) Full pipeline over a synthetic 1600x2200 page
# ---------------------------------------------------------------------------


def test_full_pipeline_page_image(tmp_path) -> None:
    # 2200x1600 input -> canonical 1800x1309 page (scale 9/11).
    image_path = tmp_path / "synthetic_page.png"
    write_image(image_path, height=2200, width=1600)
    page_shape = (1800, 1309, 3)
    scale = 9 / 11  # canonical_page_max_dim / image height

    params = make_params()
    windows = slice_windows_of(1800, 1309, params)
    last_window = windows[-1]

    full_page_detections = [
        (600, 1100, 900, 1400, 0.60, 1),    # head -> parent for the mid-conf face
        (100, 300, 300, 500, 0.92, 2),      # hi-conf face -> unconditional keep
        (600, 1800, 900, 2100, 0.40, 0),    # body below 0.52 -> threshold drop
        (-20, -20, 80, 80, 0.88, 0),       # body past the corner -> clamped
        (2000, 2000, 2100, 2100, 0.95, 2),  # fully outside -> degenerate drop
        (300, 300, 300, 400, 0.95, 2),     # zero width -> degenerate drop
    ]
    last_slice_detections = [
        (28, 42, 128, 142, 0.60, 2),   # mid-conf face inside the head -> kept
        (400, 500, 500, 600, 0.58, 2),  # mid-conf face outside the head -> anatomy drop
    ]

    def handler(shape, call_index):
        if shape == page_shape:
            return full_page_detections
        if call_index == len(windows) - 1:
            return last_slice_detections
        return []

    extractor, call_shapes = scripted_extractor(handler)

    parts = extractor.extract_from_image(image_path)

    assert len(call_shapes) == len(windows) + 1  # slices + full-page pass

    def to_original(x1, y1, x2, y2):
        return tuple(int(v / scale) for v in (x1, y1, x2, y2))

    contained_face_page = (
        last_window[0] + 28,
        last_window[1] + 42,
        last_window[0] + 128,
        last_window[1] + 142,
    )
    dangling_face_page = (
        last_window[0] + 400,
        last_window[1] + 500,
        last_window[0] + 500,
        last_window[1] + 600,
    )

    expected = [
        part(*to_original(100, 300, 300, 500), 0.92, "face"),
        part(*to_original(*contained_face_page), 0.60, "face"),
        part(*to_original(600, 1100, 900, 1400), 0.60, "head"),
        part(0, 0, 97, 97, 0.88, "body"),
    ]
    assert parts == expected

    # Explicit negative-space checks for each pruned/filtered branch.
    dangling = part(*to_original(*dangling_face_page), 0.58, "face")
    assert dangling not in parts  # mid-conf face without a containing parent
    assert all(p.score != 0.40 for p in parts)  # below body threshold
    assert all(p.x2 <= 1600 and p.y2 <= 2200 for p in parts)  # clamped
    assert len(parts) == 4


def test_full_pipeline_clamps_and_prunes(tmp_path) -> None:
    image_path = tmp_path / "edge_panel.png"
    write_image(image_path, height=100, width=100)

    def handler(_shape, _index):
        return [
            (-20, -20, 50, 50, 0.9, 0),         # clamps to (0, 0, 50, 50)
            (90, 90, 130, 120, 0.9, 0),          # clamps to (90, 90, 100, 100)
            (150, 150, 180, 190, 0.9, 0),        # fully outside -> dropped
            (40, 40, 40, 80, 0.9, 0),            # zero width -> dropped
            (float("nan"), 0, 10, 10, 0.9, 0),   # non-finite -> dropped
        ]

    extractor, _ = scripted_extractor(handler)

    parts = extractor.extract_from_image(image_path)

    assert parts == [
        part(0, 0, 50, 50, 0.9, "body"),
        part(90, 90, 100, 100, 0.9, "body"),
    ]


def test_clamp_box_to_image_pure() -> None:
    assert clamp_box_to_image((-5.2, -1.7, 50.9, 60.4), 60, 50) == (0, 0, 50, 60)
    assert clamp_box_to_image((100.0, 100.0, 200.0, 200.0), 100, 100) is None
    assert clamp_box_to_image((10.0, 10.0, 10.0, 20.0), 100, 100) is None
    assert clamp_box_to_image((float("inf"), 0.0, 1.0, 1.0), 10, 10) is None


# ---------------------------------------------------------------------------
# Custom SahiParams change behavior (config-driven contract)
# ---------------------------------------------------------------------------


def test_custom_panel_max_dim_changes_gating(tmp_path) -> None:
    # A 900px image is a sliced page under the default 800 gate, but a
    # single-pass panel under a custom 2000 gate.
    image_path = tmp_path / "fine.png"
    write_image(image_path, height=900, width=900)

    default_extractor, default_calls = scripted_extractor(lambda _s, _i: [])
    custom_extractor, custom_calls = scripted_extractor(
        lambda _s, _i: [], panel_max_dim=2000
    )

    default_extractor.extract_from_image(image_path)
    custom_extractor.extract_from_image(image_path)

    assert len(default_calls) > 1  # sliced page: windows + full-page pass
    assert len(custom_calls) == 1   # direct panel pass
    assert custom_calls[0] == (900, 900, 3)


def test_custom_class_thresholds_change_filtering(tmp_path) -> None:
    image_path = tmp_path / "panel.png"
    write_image(image_path, height=700, width=700)
    boxes = [(10, 10, 100, 100, 0.90, 2)]  # dangling face at 0.9

    strict_hier, _ = scripted_extractor(
        lambda _s, _i: boxes,
        class_conf_thresholds={"body": 0.52, "head": 0.52, "face": 0.99},
    )
    strict_flat, _ = scripted_extractor(
        lambda _s, _i: boxes,
        class_conf_thresholds={"body": 0.52, "head": 0.52, "face": 0.99},
        hierarchical_verify=False,
    )
    default_extractor, _ = scripted_extractor(lambda _s, _i: boxes)

    # With the hierarchy on, faces are governed by its banded protocol:
    # 0.9 >= 0.70 close-up band is unconditionally retained, so a custom
    # face threshold cannot drop it (spec §2).
    assert strict_hier.extract_from_image(image_path) == [
        part(10, 10, 100, 100, 0.9, "face")
    ]
    # With the hierarchy disabled the class threshold governs faces.
    assert strict_flat.extract_from_image(image_path) == []
    assert default_extractor.extract_from_image(image_path) == [
        part(10, 10, 100, 100, 0.9, "face")
    ]


def test_custom_hierarchy_disabled_keeps_dangling_face(tmp_path) -> None:
    image_path = tmp_path / "panel.png"
    write_image(image_path, height=700, width=700)
    boxes = [(10, 10, 100, 100, 0.58, 2)]  # mid-band face, no parent

    strict_extractor, _ = scripted_extractor(lambda _s, _i: boxes)
    lax_extractor, _ = scripted_extractor(lambda _s, _i: boxes, hierarchical_verify=False)

    assert strict_extractor.extract_from_image(image_path) == []  # anatomy drop
    assert lax_extractor.extract_from_image(image_path) == [
        part(10, 10, 100, 100, 0.58, "face")
    ]


def test_custom_nmm_postprocess_keeps_largest_box(tmp_path) -> None:
    image_path = tmp_path / "panel.png"
    write_image(image_path, height=700, width=700)
    overlapping = [(10, 10, 100, 100, 0.9, 0), (12, 12, 102, 102, 0.6, 0)]

    wbf_extractor, _ = scripted_extractor(lambda _s, _i: overlapping)
    nmm_extractor, _ = scripted_extractor(
        lambda _s, _i: overlapping, postprocess_type="NMM"
    )

    wbf_parts = wbf_extractor.extract_from_image(image_path)
    nmm_parts = nmm_extractor.extract_from_image(image_path)

    # WBF averages coordinates and scores; NMM keeps the biggest member.
    # DetectedPart coordinates are int-truncated after fusion (10.8 -> 10).
    assert wbf_parts[0].x1 == int((0.9 * 10 + 0.6 * 12) / 1.5)
    assert wbf_parts[0].x2 == int((0.9 * 100 + 0.6 * 102) / 1.5)
    assert wbf_parts[0].score == pytest.approx(0.75)
    assert nmm_parts == [part(10, 10, 100, 100, 0.9, "body")]


# ---------------------------------------------------------------------------
# Fail-loud validation at the boundaries
# ---------------------------------------------------------------------------


def test_init_rejects_wrong_sahi_params_type() -> None:
    with pytest.raises(TypeError, match="SahiParams"):
        SahiExtractor("m.pt", sahi_params={"slice_height": 640})  # type: ignore[arg-type]


def test_init_rejects_non_callable_predictor() -> None:
    with pytest.raises(TypeError, match="callable"):
        SahiExtractor("m.pt", predictor="not-callable")  # type: ignore[arg-type]


def test_pipeline_fails_loud_on_unknown_model_class_id(tmp_path) -> None:
    image_path = tmp_path / "panel.png"
    write_image(image_path, height=700, width=700)
    extractor, _ = scripted_extractor(lambda _s, _i: [(0, 0, 10, 10, 0.9, 7)])

    with pytest.raises(ValueError, match="unknown class id 7"):
        extractor.extract_from_image(image_path)


def test_pipeline_fails_loud_on_malformed_predictor_output(tmp_path) -> None:
    image_path = tmp_path / "panel.png"
    write_image(image_path, height=700, width=700)

    for bad_output in ([(1, 2, 3)], [(1, 2, 3, 4, 5, "six")]):
        extractor, _ = scripted_extractor(lambda _s, _i: bad_output)
        with pytest.raises(ValueError, match="Predictor item"):
            extractor.extract_from_image(image_path)


# ---------------------------------------------------------------------------
# (f) extract_from_directory
# ---------------------------------------------------------------------------


def build_directory_fixture(tmp_path) -> tuple[Path, callable, list[Path]]:
    """Panel images with per-shape scripted detections plus noise files."""
    input_dir = tmp_path / "pages"
    nested = input_dir / "nested"
    nested.mkdir(parents=True)

    stems = [
        input_dir / "alpha.png",
        input_dir / "beta.png",
        nested / "gamma.png",
    ]
    write_image(stems[0], height=600, width=600)
    write_image(stems[1], height=700, width=700)
    write_image(stems[2], height=640, width=640)

    (input_dir / "notes.txt").write_text("not an image")
    (input_dir / "corrupt.png").write_bytes(b"this is not a real png")

    detections_by_shape = {
        (600, 600, 3): [(10, 10, 100, 100, 0.9, 0)],
        (700, 700, 3): [(0, 0, 150, 150, 0.90, 2), (50, 50, 250, 250, 0.60, 1)],
        (640, 640, 3): [(5, 5, 50, 55, 0.60, 1)],
    }

    def handler(shape, _index):
        return detections_by_shape.get(shape, [])

    return input_dir, handler, stems


def test_extract_from_directory_saves_crops_and_summary(tmp_path) -> None:
    input_dir, handler, stems = build_directory_fixture(tmp_path)
    output_dir = tmp_path / "crops"
    extractor, _ = scripted_extractor(handler)

    summary = extractor.extract_from_directory(input_dir, output_dir, save_crops=True)

    assert isinstance(summary, ExtractionSummary)
    assert summary.images_processed == 3  # corrupt.png is skipped
    assert summary.crops_saved == 4
    assert len(summary.crops) == 4

    expected_crops = [
        (str(stems[0]), "alpha_crop_0.jpg", [10, 10, 100, 100], "body", 0.9),
        (str(stems[1]), "beta_crop_0.jpg", [0, 0, 150, 150], "face", 0.90),
        (str(stems[1]), "beta_crop_1.jpg", [50, 50, 250, 250], "head", 0.60),
        (str(stems[2]), "gamma_crop_0.jpg", [5, 5, 50, 55], "head", 0.60),
    ]
    for entry, (image, filename, bbox, class_name, score) in zip(
        summary.crops, expected_crops
    ):
        assert entry["image"] == image
        assert entry["crop_path"] is not None
        assert entry["crop_path"].endswith(filename)
        assert entry["bbox"] == bbox
        assert entry["class_name"] == class_name
        assert entry["score"] == score
        assert Path(entry["crop_path"]).is_file()

    saved = sorted(p.name for p in output_dir.glob("*.jpg"))
    assert saved == sorted(name for _, name, _, _, _ in expected_crops)


def test_extract_from_directory_without_save_crops(tmp_path) -> None:
    input_dir, handler, stems = build_directory_fixture(tmp_path)
    output_dir = tmp_path / "no_crops"
    extractor, _ = scripted_extractor(handler)

    summary = extractor.extract_from_directory(input_dir, output_dir, save_crops=False)

    assert summary.images_processed == 3
    assert summary.crops_saved == 0
    assert len(summary.crops) == 4  # detections still reported, no crop files
    assert all(entry["crop_path"] is None for entry in summary.crops)
    assert {entry["image"] for entry in summary.crops} == {str(s) for s in stems}
    assert not output_dir.exists()  # not created when nothing is saved


def test_extract_from_directory_missing_input_fails_loud(tmp_path) -> None:
    extractor, _ = scripted_extractor(lambda _s, _i: [])

    with pytest.raises(NotADirectoryError):
        extractor.extract_from_directory(tmp_path / "missing", tmp_path / "out")


def test_find_image_files_scans_recursively_and_ignores_noise(tmp_path) -> None:
    input_dir, _handler, _stems = build_directory_fixture(tmp_path)
    deep = input_dir / "nested" / "deep"
    deep.mkdir()
    write_image(deep / "delta.webp", height=500, width=500)

    found = find_image_files(input_dir)

    assert [p.name for p in found] == [
        "alpha.png",
        "beta.png",
        "corrupt.png",
        "delta.webp",
        "gamma.png",
    ]


def test_find_image_files_is_case_insensitive(tmp_path) -> None:
    scan_dir = tmp_path / "scan"
    scan_dir.mkdir()
    write_image(scan_dir / "UPPER.PNG", height=10, width=10)

    found = find_image_files(scan_dir)

    assert [p.name for p in found] == ["UPPER.PNG"]


def test_find_image_files_empty_directory(tmp_path) -> None:
    (tmp_path / "empty").mkdir()

    assert find_image_files(tmp_path / "empty") == []


def test_find_image_files_missing_directory_fails_loud(tmp_path) -> None:
    with pytest.raises(NotADirectoryError):
        find_image_files(tmp_path / "missing")
