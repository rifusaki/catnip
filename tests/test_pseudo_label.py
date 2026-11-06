"""Synthetic tests for the pseudo-labeling pipeline (scripts/unify/pseudo_label.py).

Teacher inference is fully replaced by crafted ``Detection`` outputs, and the
staging trees are fakes under ``tmp_path``: no images are decoded, no model
weights load, no network. All geometry is asserted on synthetic boxes.
"""

import json
import os
from pathlib import Path

import pytest

from scripts.unify import pseudo_label
from scripts.unify.pseudo_label import (
    DOWNSAMPLE_KEEP_PROB,
    downsample_headshots,
    verify_anime_bodies,
    verify_anime_faces,
    verify_manga109_heads,
    weighted_box_fusion,
)
from src.eval.adapters import Detection
from src.preprocess.box_calibration import FaceBoxAdapter


def det(x1: float, y1: float, x2: float, y2: float, score: float = 0.9, class_id: int = 0) -> Detection:
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=score, class_id=class_id)


# ========================================================================
# Weighted Box Fusion
# ========================================================================

def test_wbf_fuses_overlapping_boxes_with_score_weighted_coordinates() -> None:
    first = det(0.40, 0.40, 0.60, 0.60, score=0.9)
    second = det(0.44, 0.44, 0.64, 0.64, score=0.7)

    fused = weighted_box_fusion([first, second])

    assert len(fused) == 1
    box = fused[0]
    assert box.x1 == pytest.approx((0.9 * 0.40 + 0.7 * 0.44) / 1.6)
    assert box.y1 == pytest.approx((0.9 * 0.40 + 0.7 * 0.44) / 1.6)
    assert box.x2 == pytest.approx((0.9 * 0.60 + 0.7 * 0.64) / 1.6)
    assert box.y2 == pytest.approx((0.9 * 0.60 + 0.7 * 0.64) / 1.6)
    assert box.score == pytest.approx(0.8)  # member mean confidence
    assert box.class_id == 0  # from the highest-scored member


def test_wbf_keeps_disjoint_boxes_apart() -> None:
    left = det(0.00, 0.00, 0.20, 0.20, score=0.5)
    right = det(0.80, 0.80, 1.00, 1.00, score=0.9)

    fused = weighted_box_fusion([left, right])

    assert len(fused) == 2
    assert [box.score for box in fused] == [0.9, 0.5]  # score-descending output


def test_wbf_fuses_nested_boxes() -> None:
    outer = det(0.30, 0.30, 0.50, 0.50, score=0.8)
    inner = det(0.35, 0.35, 0.45, 0.45, score=0.6)

    assert len(weighted_box_fusion([outer, inner])) == 1


def test_wbf_empty_and_deterministic() -> None:
    assert weighted_box_fusion([]) == []

    first = det(0.40, 0.40, 0.60, 0.60, score=0.9)
    second = det(0.41, 0.42, 0.62, 0.63, score=0.6)
    assert weighted_box_fusion([first, second]) == weighted_box_fusion([second, first])


# ========================================================================
# verify_manga109_heads (Rule A + Rule B)
# ========================================================================

def test_rule_a_accepts_and_clamps_around_gt_face() -> None:
    face = det(0.40, 0.40, 0.50, 0.50, score=1.0, class_id=2)
    teacher_head = det(0.405, 0.405, 0.51, 0.51, score=0.8, class_id=1)

    accepted = verify_manga109_heads([teacher_head], [], [face], [])

    assert len(accepted) == 1
    head = accepted[0]
    # Clamped outward to fully cover the face.
    assert (head.x1, head.y1) == (pytest.approx(0.40), pytest.approx(0.40))
    assert (head.x2, head.y2) == (pytest.approx(0.51), pytest.approx(0.51))
    assert head.class_id == 1


def test_rule_a_rejects_disjoint_head() -> None:
    face = det(0.40, 0.40, 0.50, 0.50, score=1.0, class_id=2)
    elsewhere = det(0.05, 0.05, 0.12, 0.12, score=0.9, class_id=1)

    assert verify_manga109_heads([elsewhere], [], [face], []) == []


def test_rule_a_rejects_oversized_and_undersized_heads() -> None:
    face = det(0.40, 0.40, 0.50, 0.50, score=1.0, class_id=2)
    oversized = det(0.05, 0.05, 0.95, 0.95, score=0.9, class_id=1)  # ratio ~81 > 4.5
    undersized = det(0.40, 0.40, 0.505, 0.505, score=0.9, class_id=1)  # ratio 1.10 < 1.15

    assert verify_manga109_heads([oversized], [], [face], []) == []
    assert verify_manga109_heads([undersized], [], [face], []) == []


def test_rule_b_accepts_high_confidence_head_on_faceless_body() -> None:
    body = det(0.30, 0.20, 0.70, 0.80, score=1.0, class_id=0)
    # Fully inside the top-35% band ([0.20, 0.41] vertically) and inside
    # the body's x-extent: the plan's "H lies within the top 35% of B".
    back_of_head = det(0.42, 0.20, 0.58, 0.40, score=0.65, class_id=1)

    accepted = verify_manga109_heads([back_of_head], [], [], [body])

    assert len(accepted) == 1
    assert accepted[0].class_id == 1


def test_rule_b_rejects_low_confidence_or_misplaced_heads() -> None:
    body = det(0.30, 0.20, 0.70, 0.80, score=1.0, class_id=0)

    low_conf = det(0.42, 0.20, 0.58, 0.40, score=0.60, class_id=1)
    below_band = det(0.42, 0.42, 0.58, 0.62, score=0.90, class_id=1)  # center 0.52 > band 0.41
    # Center inside the band but the box POKE BELOW the band edge 0.41:
    # the plan says the head lies WITHIN the band, so center-only
    # placement does not qualify.
    pokes_below_band = det(0.42, 0.30, 0.58, 0.44, score=0.90, class_id=1)
    # Horizontally outside the body: not "within" the band region either.
    spills_x = det(0.20, 0.22, 0.36, 0.38, score=0.90, class_id=1)

    assert verify_manga109_heads([low_conf], [], [], [body]) == []
    assert verify_manga109_heads([below_band], [], [], [body]) == []
    assert verify_manga109_heads([pokes_below_band], [], [], [body]) == []
    assert verify_manga109_heads([spills_x], [], [], [body]) == []


def test_rule_b_rejects_bodies_that_already_have_a_face() -> None:
    body = det(0.30, 0.20, 0.70, 0.80, score=1.0, class_id=0)
    associated_face = det(0.42, 0.22, 0.58, 0.40, score=1.0, class_id=2)  # IoS(face, body) = 1.0
    head = det(0.43, 0.21, 0.57, 0.41, score=0.90, class_id=1)
    # Rule A also rejects: head/face area ratio ~1.0 < 1.15.

    assert verify_manga109_heads([head], [], [associated_face], [body]) == []


def test_teacher_pools_are_fused_before_verification() -> None:
    face = det(0.40, 0.40, 0.50, 0.50, score=1.0, class_id=2)
    teacher_head = det(0.405, 0.405, 0.51, 0.51, score=0.8, class_id=1)
    catnip_head = det(0.39, 0.39, 0.53, 0.53, score=0.6, class_id=1)

    accepted = verify_manga109_heads([teacher_head], [catnip_head], [face], [])

    assert len(accepted) == 1  # near-duplicate boxes collapse into one fused head
    head = accepted[0]
    expected_x1 = (0.8 * 0.405 + 0.6 * 0.39) / 1.4
    expected_x2 = (0.8 * 0.51 + 0.6 * 0.53) / 1.4
    assert head.x1 == pytest.approx(expected_x1)
    assert head.x2 == pytest.approx(expected_x2)
    assert head.y1 == pytest.approx(expected_x1)  # symmetric construction
    assert head.y2 == pytest.approx(expected_x2)
    assert head.score == pytest.approx(0.7)  # fused confidence = member mean


# ========================================================================
# verify_anime_faces
# ========================================================================

def test_anime_faces_keeps_one_transformed_face_per_head() -> None:
    head = det(0.30, 0.30, 0.60, 0.60, score=1.0, class_id=1)
    low_score_face = det(0.35, 0.35, 0.50, 0.50, score=0.7)
    high_score_face = det(0.36, 0.34, 0.51, 0.49, score=0.9)
    adapter = FaceBoxAdapter(alpha_x=0.02, alpha_y=-0.04, beta_w=0.8, beta_h=0.75)

    accepted = verify_anime_faces([low_score_face, high_score_face], [head], adapter)

    assert len(accepted) == 1  # at most one face per head, highest score wins
    face = accepted[0]
    assert face.class_id == 2
    assert face.score == pytest.approx(0.9)
    # (0.435, 0.415, 0.15, 0.15) transformed by the adapter:
    assert (face.x1, face.y1, face.x2, face.y2) == (
        pytest.approx(0.378), pytest.approx(0.35275), pytest.approx(0.498), pytest.approx(0.46525)
    )


def test_anime_faces_require_strict_containment_and_confidence() -> None:
    head = det(0.30, 0.30, 0.60, 0.60, score=1.0, class_id=1)
    adapter = FaceBoxAdapter(alpha_x=0.0, alpha_y=0.0, beta_w=1.0, beta_h=1.0)

    straddling = det(0.20, 0.30, 0.50, 0.60, score=0.9)  # IoS 0.667 < 0.85
    weak = det(0.35, 0.35, 0.50, 0.50, score=0.30)  # below conf_face

    assert verify_anime_faces([straddling], [head], adapter) == []
    assert verify_anime_faces([weak], [head], adapter) == []


def test_anime_faces_independent_per_head() -> None:
    adapter = FaceBoxAdapter(alpha_x=0.0, alpha_y=0.0, beta_w=1.0, beta_h=1.0)
    heads = [
        det(0.10, 0.10, 0.30, 0.30, score=1.0, class_id=1),
        det(0.60, 0.60, 0.90, 0.90, score=1.0, class_id=1),
    ]
    faces = [det(0.15, 0.15, 0.25, 0.25, score=0.9), det(0.65, 0.65, 0.80, 0.80, score=0.8)]

    accepted = verify_anime_faces(faces, heads, adapter)
    assert len(accepted) == 2
    centers = sorted((box.x1, box.y1) for box in accepted)
    assert centers[0] == (pytest.approx(0.15), pytest.approx(0.15))
    assert centers[1] == (pytest.approx(0.65), pytest.approx(0.65))


# ========================================================================
# verify_anime_bodies
# ========================================================================

def test_bodies_accept_full_figure_crops() -> None:
    head = det(0.45, 0.10, 0.55, 0.30, score=1.0, class_id=1)
    body = det(0.20, 0.10, 0.80, 0.90, score=0.60, class_id=0)

    accepted = verify_anime_bodies([body], [head])
    assert len(accepted) == 1
    assert accepted[0].class_id == 0


def test_bodies_reject_headshot_rectangles_and_partial_heads() -> None:
    head = det(0.45, 0.10, 0.55, 0.30, score=1.0, class_id=1)

    barely_bigger_than_head = det(0.445, 0.095, 0.555, 0.305, score=0.9)  # IoU ~0.81 > 0.80
    misses_head = det(0.50, 0.10, 0.90, 0.50, score=0.9)  # IoS(head, body) ~0.03 < 0.70

    assert verify_anime_bodies([barely_bigger_than_head], [head]) == []
    assert verify_anime_bodies([misses_head], [head]) == []


def test_bodies_confidence_floor() -> None:
    head = det(0.45, 0.10, 0.55, 0.30, score=1.0, class_id=1)
    weak_full_figure = det(0.20, 0.10, 0.80, 0.90, score=0.40, class_id=0)

    assert verify_anime_bodies([weak_full_figure], [head]) == []


# ========================================================================
# Deterministic headshot downsampling
# ========================================================================

def test_downsample_headshots_is_deterministic_and_order_independent() -> None:
    names = [f"img_{i:03d}.jpg" for i in range(40)]
    counts = {name: 0 for name in names}

    first = downsample_headshots(names, counts, DOWNSAMPLE_KEEP_PROB, seed=42)
    second = downsample_headshots(names, counts, DOWNSAMPLE_KEEP_PROB, seed=42)
    shuffled = downsample_headshots(list(reversed(names)), counts, DOWNSAMPLE_KEEP_PROB, seed=42)

    assert first == second
    assert first == shuffled
    assert 0 < len(first) < len(names)  # a real draw, not all-or-nothing


def test_downsample_headshots_keeps_images_with_bodies_unconditionally() -> None:
    names = ["figure.jpg", "headshot.jpg"]
    counts = {"figure.jpg": 1, "headshot.jpg": 0}

    assert downsample_headshots(names, counts, keep_prob=0.0, seed=42) == {"figure.jpg"}
    assert downsample_headshots(names, counts, keep_prob=1.0, seed=42) == {"figure.jpg", "headshot.jpg"}


def test_downsample_headshots_decisions_are_independent_per_filename() -> None:
    names = ["a.png", "b.jpg", "c.webp"]
    counts = {name: 0 for name in names}

    for name in names:
        solo = downsample_headshots([name], counts, DOWNSAMPLE_KEEP_PROB, seed=42)
        grouped = downsample_headshots(names, counts, DOWNSAMPLE_KEEP_PROB, seed=42)
        assert (name in grouped) == (name in solo)


def test_downsample_headshots_rejects_invalid_probability() -> None:
    with pytest.raises(ValueError, match="keep_prob"):
        downsample_headshots(["a.png"], {"a.png": 0}, keep_prob=1.5, seed=42)


# ========================================================================
# Staging-tree writer & CLI (monkeypatched teacher runner)
# ========================================================================

def write_file(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def build_manga109_staging(root: Path) -> Path:
    staging = root / "staging" / "manga109"
    for name in ("page_001.jpg", "page_002.jpg"):
        write_file(staging / "images" / name, "fake-image-bytes")
    write_file(
        staging / "labels" / "page_001.txt",
        "2 0.45 0.45 0.10 0.10\n0 0.50 0.55 0.30 0.40\n",
    )
    write_file(
        staging / "labels" / "page_002.txt",
        "2 0.45 0.45 0.10 0.10\n0 0.50 0.55 0.30 0.40\n",
    )
    return root / "staging"


def build_anime_staging(root: Path) -> Path:
    staging = root / "staging" / "ah_yolo"
    for index in range(1, 7):
        write_file(staging / "images" / f"art_{index:03d}.jpg", "fake-image-bytes")
        write_file(staging / "labels" / f"art_{index:03d}.txt", "1 0.5 0.5 0.2 0.2\n")
    # ah_coco belongs to the anime section too but is intentionally empty.
    (root / "staging" / "ah_coco" / "images").mkdir(parents=True, exist_ok=True)
    (root / "staging" / "ah_coco" / "labels").mkdir(parents=True, exist_ok=True)
    return root / "staging"


MANGA109_TEACHERS = {
    "deepghs_head_x": {
        "page_001.jpg": [det(0.405, 0.405, 0.51, 0.51, score=0.8, class_id=1)],
        "page_002.jpg": [det(0.05, 0.05, 0.12, 0.12, score=0.9, class_id=1)],  # accepts nothing
    },
    "catnip": {
        "page_001.jpg": [det(0.39, 0.39, 0.53, 0.53, score=0.6, class_id=1)],
        "page_002.jpg": [],
    },
}


ANIME_TEACHERS = {
    "deepghs_face_s": {
        # face predictions sit strictly inside the GT head (0.40-0.60),
        # cxcywh (0.5, 0.5, 0.12, 0.12):
        #   adapter(0.02, -0.04, 0.8, 0.75) -> (0.5024, 0.4952, 0.096, 0.09)
        #   -> line "2 0.502400 0.495200 0.096000 0.090000".
        "art_001.jpg": [det(0.44, 0.44, 0.56, 0.56, score=0.9)],
        "art_002.jpg": [det(0.44, 0.44, 0.56, 0.56, score=0.9)],
        "art_006.jpg": [det(0.44, 0.44, 0.56, 0.56, score=0.9)],
    },
    "zymk9_face": {},
    "deepghs_person_s": {
        "art_001.jpg": [det(0.20, 0.10, 0.80, 0.90, score=0.70)],
    },
}


def patch_teachers(monkeypatch: pytest.MonkeyPatch, predictions: dict) -> None:
    def fake_run_teachers(image_path, model_key, target, conf, device="cpu", custom_path=None):
        return predictions.get(model_key, {}).get(image_path.name, [])

    monkeypatch.setattr(pseudo_label, "run_teachers", fake_run_teachers)


def test_manga109_writer_enriches_labels_and_preserves_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = build_manga109_staging(tmp_path)
    patch_teachers(monkeypatch, MANGA109_TEACHERS)
    output = tmp_path / "enriched"

    originals = {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted((staging / "manga109" / "labels").glob("*.txt"))
    }

    pseudo_label.main(
        [
            "--target-section", "manga109",
            "--staging-dir", str(staging),
            "--output-dir", str(output),
            "--catnip-model", str(tmp_path / "catnip.pt"),
        ]
    )

    # Labels: pristine original lines + the verified fused synthetic head.
    expected_001 = (
        "2 0.45 0.45 0.10 0.10\n"
        "0 0.50 0.55 0.30 0.40\n"
        "1 0.458571 0.458571 0.120000 0.120000\n"  # WBF of the two teacher heads, clamped over the face
    )
    label_001 = (output / "manga109" / "labels" / "page_001.txt").read_text(encoding="utf-8")
    assert label_001 == expected_001

    label_002 = (output / "manga109" / "labels" / "page_002.txt").read_text(encoding="utf-8")
    assert label_002 == "2 0.45 0.45 0.10 0.10\n0 0.50 0.55 0.30 0.40\n"  # untouched + nothing synthetic

    # Images are valid symlinks back into the pristine staging tree.
    for name in ("page_001.jpg", "page_002.jpg"):
        link = output / "manga109" / "images" / name
        assert link.is_symlink()
        assert link.resolve() == (staging / "manga109" / "images" / name).resolve()

    # Manifest reflects per-image decisions.
    manifest = json.loads((output / "manga109" / "manifest.json").read_text(encoding="utf-8"))
    entry_001 = manifest["images"]["page_001.jpg"]
    assert entry_001["kept"] is True
    assert entry_001["reason"] == "enriched"
    assert entry_001["added"] == {"1": 1}
    assert entry_001["bodies_found"] == 1
    assert entry_001["downsampled"] is False
    assert manifest["images"]["page_002.jpg"]["added"] == {"1": 0}
    assert manifest["totals"]["images_seen"] == 2
    assert manifest["totals"]["images_kept"] == 2
    assert manifest["totals"]["added_boxes"] == {"0": 0, "1": 1, "2": 0}

    # Staging inputs were never mutated.
    for path in (staging / "manga109" / "labels").glob("*.txt"):
        assert path.read_text(encoding="utf-8") == originals[path.name]


def test_manga109_writer_rerun_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = build_manga109_staging(tmp_path)
    patch_teachers(monkeypatch, MANGA109_TEACHERS)
    output = tmp_path / "enriched"
    argv = [
        "--target-section", "manga109",
        "--staging-dir", str(staging),
        "--output-dir", str(output),
        "--catnip-model", str(tmp_path / "catnip.pt"),
    ]

    pseudo_label.main(argv)
    snapshot = {
        path.name: path.read_text(encoding="utf-8")
        for path in set(output.rglob("*.txt")) | set(output.rglob("*.json"))
    }

    pseudo_label.main(argv + ["--override"])  # reruns require --override

    for path in output.rglob("*.txt"):
        assert path.read_text(encoding="utf-8") == snapshot[path.name]
    manifest = json.loads((output / "manga109" / "manifest.json").read_text(encoding="utf-8"))
    assert json.dumps(manifest, sort_keys=True) == json.dumps(
        json.loads(snapshot["manifest.json"]), sort_keys=True
    )


def test_manga109_writer_requires_override_for_non_empty_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = build_manga109_staging(tmp_path)
    patch_teachers(monkeypatch, MANGA109_TEACHERS)
    output = tmp_path / "enriched"
    argv = [
        "--target-section", "manga109",
        "--staging-dir", str(staging),
        "--output-dir", str(output),
        "--catnip-model", str(tmp_path / "catnip.pt"),
    ]

    pseudo_label.main(argv)
    with pytest.raises(SystemExit, match="--override"):
        pseudo_label.main(argv)  # non-empty output without --override must refuse


def test_anime_writer_downsamples_headshots_and_excludes_dropped_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = build_anime_staging(tmp_path)
    patch_teachers(monkeypatch, ANIME_TEACHERS)
    output = tmp_path / "enriched"
    adapter_path = write_file(tmp_path / "adapter.json", json.dumps(
        {"alpha_x": 0.02, "alpha_y": -0.04, "beta_w": 0.8, "beta_h": 0.75}
    ))

    pseudo_label.main(
        [
            "--target-section", "anime",
            "--staging-dir", str(staging),
            "--output-dir", str(output),
            "--adapter", str(adapter_path),
            "--seed", "7",
        ]
    )

    # art_001 has a verified body; seed 7 keeps art_002 and art_006 from the
    # headshot draw; art_003/004/005 are excluded from the tree entirely.
    present = sorted(path.name for path in (output / "ah_yolo" / "images").iterdir())
    assert present == ["art_001.jpg", "art_002.jpg", "art_006.jpg"]
    labels = sorted(path.name for path in (output / "ah_yolo" / "labels").glob("*.txt"))
    assert labels == ["art_001.txt", "art_002.txt", "art_006.txt"]
    for excluded in ("art_003.jpg", "art_004.jpg", "art_005.jpg"):
        assert not (output / "ah_yolo" / "images" / excluded).exists()
        assert not (output / "ah_yolo" / "labels" / excluded.replace(".jpg", ".txt")).exists()

    # Full-figure image: original head + synthetic face + synthetic body.
    label_001 = (output / "ah_yolo" / "labels" / "art_001.txt").read_text(encoding="utf-8")
    assert label_001 == (
        "1 0.5 0.5 0.2 0.2\n"
        "2 0.502400 0.495200 0.096000 0.090000\n"  # adapter-transformed face
        "0 0.500000 0.500000 0.600000 0.800000\n"  # verified body
    )

    # Kept headshot keeps its synthetic face line (proves the exclusions above
    # were the downsample decision, not missing face predictions).
    label_002 = (output / "ah_yolo" / "labels" / "art_002.txt").read_text(encoding="utf-8")
    assert label_002 == "1 0.5 0.5 0.2 0.2\n2 0.502400 0.495200 0.096000 0.090000\n"

    manifest = json.loads((output / "ah_yolo" / "manifest.json").read_text(encoding="utf-8"))
    by_name = manifest["images"]
    assert by_name["art_001.jpg"]["reason"] == "enriched"
    assert by_name["art_001.jpg"]["bodies_found"] == 1
    assert by_name["art_001.jpg"]["downsampled"] is False
    assert by_name["art_002.jpg"]["reason"] == "headshot_kept"
    assert by_name["art_002.jpg"]["downsampled"] is True
    assert by_name["art_003.jpg"]["kept"] is False
    assert by_name["art_003.jpg"]["reason"] == "headshot_excluded"
    assert by_name["art_003.jpg"]["downsampled"] is True

    # The kept set matches the pure downsample function exactly.
    headshots = [f"art_{i:03d}.jpg" for i in range(2, 7)]
    expected = downsample_headshots(headshots, {name: 0 for name in headshots}, seed=7)
    assert expected == {"art_002.jpg", "art_006.jpg"}

    totals = manifest["totals"]
    assert totals["images_seen"] == 6
    assert totals["images_kept"] == 3
    assert totals["images_excluded"] == 3
    assert totals["headshot_only"] == 5
    assert totals["headshot_kept"] == 2
    assert totals["added_boxes"] == {"0": 1, "1": 0, "2": 3}  # only from written images


def test_dry_run_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    staging = build_anime_staging(tmp_path)
    patch_teachers(monkeypatch, ANIME_TEACHERS)
    adapter_path = write_file(tmp_path / "adapter.json", json.dumps(
        {"alpha_x": 0.02, "alpha_y": -0.04, "beta_w": 0.8, "beta_h": 0.75}
    ))
    output = tmp_path / "enriched"

    pseudo_label.main(
        [
            "--target-section", "anime",
            "--staging-dir", str(staging),
            "--output-dir", str(output),
            "--adapter", str(adapter_path),
            "--dry-run",
        ]
    )

    assert not output.exists()


def test_limit_caps_images_per_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    staging = build_manga109_staging(tmp_path)
    patch_teachers(monkeypatch, MANGA109_TEACHERS)
    output = tmp_path / "limit_enriched"

    pseudo_label.main(
        [
            "--target-section", "manga109",
            "--staging-dir", str(staging),
            "--output-dir", str(output),
            "--catnip-model", str(tmp_path / "catnip.pt"),
            "--limit", "1",
        ]
    )

    manifest = json.loads((output / "manga109" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["totals"]["images_seen"] == 1
    assert list(manifest["images"]) == ["page_001.jpg"]


def test_anime_section_requires_adapter(tmp_path: Path) -> None:
    staging = build_anime_staging(tmp_path)
    with pytest.raises(SystemExit) as err:
        pseudo_label.main(
            [
                "--target-section", "anime",
                "--staging-dir", str(staging),
                "--output-dir", str(tmp_path / "out"),
            ]
        )
    assert err.value.code == 2


def test_manga109_section_requires_catnip_model(tmp_path: Path) -> None:
    staging = build_manga109_staging(tmp_path)
    with pytest.raises(SystemExit) as err:
        pseudo_label.main(
            [
                "--target-section", "manga109",
                "--staging-dir", str(staging),
                "--output-dir", str(tmp_path / "out"),
            ]
        )
    assert err.value.code == 2


def test_missing_staging_source_fails_loud(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    staging = build_anime_staging(tmp_path)  # manga109/ is intentionally absent
    patch_teachers(monkeypatch, ANIME_TEACHERS)
    adapter_path = write_file(tmp_path / "adapter.json", json.dumps(
        {"alpha_x": 0.02, "alpha_y": -0.04, "beta_w": 0.8, "beta_h": 0.75}
    ))

    with pytest.raises(SystemExit) as err:
        pseudo_label.main(
            [
                "--target-section", "all",
                "--staging-dir", str(staging),
                "--output-dir", str(tmp_path / "out"),
                "--adapter", str(adapter_path),
                "--catnip-model", str(tmp_path / "catnip.pt"),
            ]
        )
    assert err.value.code == 1


def test_symlink_targets_are_absolute_and_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = build_manga109_staging(tmp_path)
    patch_teachers(monkeypatch, MANGA109_TEACHERS)
    output = tmp_path / "enriched"

    pseudo_label.main(
        [
            "--target-section", "manga109",
            "--staging-dir", str(staging),
            "--output-dir", str(output),
            "--catnip-model", str(tmp_path / "catnip.pt"),
        ]
    )

    for link in (output / "manga109" / "images").iterdir():
        assert os.path.islink(link)
        target = Path(os.readlink(link))
        assert target.is_absolute()
        assert target.exists()


def test_manga109_writer_removes_stale_files_from_prior_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--override reruns converge to the kept set (review M4 regression).

    Files left in images/labels by an earlier run (e.g. a headshot that a
    different --seed now excludes) must be removed: leaking them into
    unify-stage1 would silently contradict the manifest's EXCLUDED verdicts.
    """
    staging = build_manga109_staging(tmp_path)
    patch_teachers(monkeypatch, MANGA109_TEACHERS)
    output = tmp_path / "enriched"
    argv = [
        "--target-section", "manga109",
        "--staging-dir", str(staging),
        "--output-dir", str(output),
        "--catnip-model", str(tmp_path / "catnip.pt"),
    ]

    pseudo_label.main(argv)

    # Leftovers from a hypothetical prior run with a different kept set.
    stale_image = output / "manga109" / "images" / "ghost_from_previous_seed.jpg"
    stale_image.write_bytes(b"stale")
    stale_label = output / "manga109" / "labels" / "ghost_from_previous_seed.txt"
    stale_label.write_text("1 0.5 0.5 0.1 0.1\n", encoding="utf-8")

    pseudo_label.main(argv + ["--override"])

    assert not stale_image.exists()
    assert not stale_label.exists()
    # The current kept set is fully present after the rerun.
    assert (output / "manga109" / "images" / "page_001.jpg").is_symlink()
    assert (output / "manga109" / "labels" / "page_001.txt").is_file()


def _build_izutsumi_calibration_staging(root: Path) -> Path:
    """Izutsumi staging pages with tight GT faces at cxcywh (0.5, 0.5, 0.12, 0.16)."""
    staging = root / "staging" / "izutsumi"
    for index in range(1, 13):
        write_file(staging / "images" / f"izupage_{index:03d}.jpg", "fake-image-bytes")
        write_file(
            staging / "labels" / f"izupage_{index:03d}.txt", "2 0.5 0.5 0.12 0.16\n"
        )
    return root / "staging"


CALIBRATION_TEACHER_PRED = det(0.412, 0.394, 0.572, 0.594, score=0.9)  # loose face


def test_calibrate_fits_and_serializes_face_box_adapter(tmp_path, monkeypatch) -> None:
    """--target-section calibrate closes the orphaned-module gap: teachers ->
    pairs -> Huber fit -> validation -> JSON (plan task 3.2 end to end)."""
    staging = _build_izutsumi_calibration_staging(tmp_path)
    build_manga109_staging(tmp_path)  # adds staging/manga109 (page_001/002)

    # Predictions are the TRUE adapter pre-images of the GT faces for
    # (alpha_x, alpha_y, beta_w, beta_h) = (0.05, 0.03, 0.75, 0.80):
    # izutsumi GT (0.5,0.5,0.12,0.16) <- pred cxcywh (0.492,0.494,0.16,0.20)
    #   -> xyxy (0.412, 0.394, 0.572, 0.594), IoU 0.6 >= 0.45.
    izutsumi_preds = {
        f"izupage_{index:03d}.jpg": [det(0.412, 0.394, 0.572, 0.594, score=0.9)]
        for index in range(1, 13)
    }
    manga109_preds = {
        # GT (0.45,0.45,0.10,0.10) <- pred cxcywh (0.443333,0.44625,0.133333,0.125)
        "page_001.jpg": [det(0.376667, 0.383750, 0.510000, 0.508750, score=0.9)],
        "page_002.jpg": [det(0.376667, 0.383750, 0.510000, 0.508750, score=0.9)],
    }
    monkeypatch.setattr(
        pseudo_label,
        "run_teachers",
        lambda image_path, model_key, target, conf, device="cpu", custom_path=None: (
            izutsumi_preds.get(image_path.name, [])
            if image_path.name.startswith("izupage")
            else manga109_preds.get(image_path.name, [])
            if model_key == "deepghs_face_s"
            else []
        ),
    )

    adapter_path = tmp_path / "face_box_adapter.json"
    pseudo_label.main(
        [
            "--target-section", "calibrate",
            "--staging-dir", str(staging),
            "--adapter", str(adapter_path),
            "--device", "cpu",
        ]
    )

    fitted = FaceBoxAdapter.from_json(adapter_path)
    assert fitted.alpha_x == pytest.approx(0.05, abs=1e-6)
    assert fitted.alpha_y == pytest.approx(0.03, abs=1e-6)
    assert fitted.beta_w == pytest.approx(0.75, abs=1e-6)
    assert fitted.beta_h == pytest.approx(0.80, abs=1e-6)

    # A transformed prediction lands exactly on the GT convention.
    xc, yc, w, h = fitted.transform((0.492, 0.494, 0.16, 0.20))
    assert (xc, yc, w, h) == pytest.approx((0.5, 0.5, 0.12, 0.16))


def test_calibrate_dry_run_writes_nothing(tmp_path, monkeypatch, capsys) -> None:
    staging = _build_izutsumi_calibration_staging(tmp_path)
    build_manga109_staging(tmp_path)

    monkeypatch.setattr(
        pseudo_label,
        "run_teachers",
        lambda image_path, model_key, target, conf, device="cpu", custom_path=None:
            [det(0.412, 0.394, 0.572, 0.594, score=0.9)]
            if image_path.name.startswith("izupage") and model_key == "deepghs_face_s"
            else [det(0.376667, 0.383750, 0.510000, 0.508750, score=0.9)]
            if model_key == "deepghs_face_s"
            else [],
    )

    adapter_path = tmp_path / "face_box_adapter.json"
    pseudo_label.main(
        [
            "--target-section", "calibrate",
            "--staging-dir", str(staging),
            "--adapter", str(adapter_path),
            "--device", "cpu",
            "--dry-run",
        ]
    )

    assert not adapter_path.exists()
    assert "DRY RUN" in capsys.readouterr().out


def test_calibrate_requires_adapter_output_path(tmp_path, capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        pseudo_label.parse_args(
            ["--target-section", "calibrate", "--staging-dir", str(tmp_path)]
        )
    assert excinfo.value.code == 2
    assert "adapter is the OUTPUT JSON path" in capsys.readouterr().err


def test_calibrate_validates_both_sources_and_reserves_fit_per_source(
    tmp_path, monkeypatch
) -> None:
    """The 80/20 split is PER SOURCE: both izutsumi and manga109 must appear
    in the validation set (final review C1) — a concatenated split would
    validate only the tail of the sorted manga109 titles."""
    staging = _build_izutsumi_calibration_staging(tmp_path)
    build_manga109_staging(tmp_path)

    monkeypatch.setattr(
        pseudo_label,
        "run_teachers",
        lambda image_path, model_key, target, conf, device="cpu", custom_path=None:
            [det(0.412, 0.394, 0.572, 0.594, score=0.9)]
            if image_path.name.startswith("izupage") and model_key == "deepghs_face_s"
            else [det(0.376667, 0.383750, 0.510000, 0.508750, score=0.9)]
            if model_key == "deepghs_face_s"
            else [],
    )

    report = pseudo_label.calibrate_face_adapter(
        staging,
        tmp_path / "adapter.json",
        device="cpu",
        limit=0,
        dry_run=True,
    )

    assert report["fit_pairs_by_source"] == {"izutsumi": 9, "manga109": 1}
    # 12 izutsumi pages -> 3 validation pairs; 2 manga109 pages -> 1.
    assert report["validation_pairs_by_source"] == {
        "izutsumi": 3,
        "manga109": 1,
    }
    assert report["validation"]["recall_at_iou_after"] == pytest.approx(1.0)


def test_calibrate_manga109_sample_is_strided_not_head_capped(tmp_path, monkeypatch) -> None:
    """--limit draws a STRIDED sample of manga109 pages (plan: a 2,000-page
    sample, not the first N pages of the first titles)."""
    staging = tmp_path / "staging" / "manga109"
    for index in range(1, 21):
        write_file(staging / "images" / f"page_{index:03d}.jpg", "fake-image-bytes")
        write_file(staging / "labels" / f"page_{index:03d}.txt", "2 0.45 0.45 0.10 0.10\n")
    (tmp_path / "staging" / "izutsumi" / "images").mkdir(parents=True, exist_ok=True)
    (tmp_path / "staging" / "izutsumi" / "labels").mkdir(parents=True, exist_ok=True)

    records = pseudo_label.collect_staging_pairs(staging, 0)
    sampled = pseudo_label._strided_sample(records, 4)

    assert len(sampled) == 4
    # Strided: pages 1, 6, 11, 16 — not the first four sorted pages.
    assert [record.image_name for record in sampled] == [
        "page_001.jpg", "page_006.jpg", "page_011.jpg", "page_016.jpg",
    ]
