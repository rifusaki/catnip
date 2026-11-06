"""Unit tests for crop linking into character instances (src/reid/dataset.py)."""

import pytest

from src.reid.dataset import CharacterInstance, link_part_detections


def _det(image, class_name, bbox, score, crop_path=None):
    return {
        "image": image,
        "crop_path": crop_path,
        "bbox": list(bbox),
        "class_name": class_name,
        "score": score,
    }


def test_nested_face_head_body_forms_one_instance() -> None:
    detections = [
        _det("page_01.png", "body", (0, 0, 100, 200), 0.9),
        _det("page_01.png", "head", (30, 20, 70, 120), 0.8),
        _det("page_01.png", "face", (40, 40, 60, 80), 0.7),
    ]

    instances = link_part_detections(detections)

    assert len(instances) == 1
    instance = instances[0]
    assert isinstance(instance, CharacterInstance)
    assert instance.image_path == "page_01.png"
    assert set(instance.parts) == {"body", "head", "face"}
    assert instance.parts["head"]["bbox"] == [30, 20, 70, 120]
    assert instance.parts["head"]["score"] == 0.8
    assert instance.parts["face"]["bbox"] == [40, 40, 60, 80]
    assert instance.instance_id == "page_01.png#001"


def test_orphan_face_becomes_its_own_instance() -> None:
    detections = [
        _det("page_01.png", "body", (0, 0, 100, 200), 0.9),
        _det("page_01.png", "head", (30, 20, 70, 120), 0.8),
        _det("page_01.png", "face", (40, 40, 60, 80), 0.7),
        _det("page_01.png", "face", (900, 900, 940, 940), 0.6),  # far away
    ]

    instances = link_part_detections(detections)

    assert len(instances) == 2
    orphan = instances[1]
    assert set(orphan.parts) == {"face"}
    assert orphan.parts["face"]["bbox"] == [900, 900, 940, 940]


def test_detections_are_grouped_per_image() -> None:
    detections = [
        _det("b_page.png", "face", (10, 10, 30, 30), 0.7),
        _det("a_page.png", "face", (10, 10, 30, 30), 0.7),
    ]

    instances = link_part_detections(detections)

    assert [instance.image_path for instance in instances] == ["a_page.png", "b_page.png"]


def test_linking_is_deterministic_under_input_shuffle() -> None:
    detections = [
        _det("page_01.png", "body", (0, 0, 100, 200), 0.9),
        _det("page_01.png", "head", (30, 20, 70, 120), 0.8),
        _det("page_01.png", "face", (40, 40, 60, 80), 0.7),
        _det("page_01.png", "body", (400, 400, 600, 900), 0.85),
        _det("page_01.png", "head", (500, 450, 560, 600), 0.66),
        _det("page_01.png", "face", (900, 900, 940, 940), 0.6),
    ]

    forward = link_part_detections(detections)
    shuffled = link_part_detections(list(reversed(detections)))

    assert forward == shuffled


def test_head_joins_best_containing_body() -> None:
    # head fully inside body1 (IoS 1.0) but only partially inside body2.
    detections = [
        _det("page.png", "body", (0, 0, 100, 200), 0.9),      # body1
        _det("page.png", "body", (40, 20, 200, 120), 0.8),     # body2 overlaps head
        _det("page.png", "head", (30, 20, 70, 120), 0.7),
    ]

    instances = link_part_detections(detections)

    # Two bodies -> two instances; the head joins body1 only.
    assert len(instances) == 2
    with_head = next(i for i in instances if "head" in i.parts)
    assert with_head.parts["body"]["bbox"] == [0, 0, 100, 200]
    headless = next(i for i in instances if "head" not in i.parts)
    assert headless.parts["body"]["bbox"] == [40, 20, 200, 120]


def test_containment_below_threshold_leaves_child_unlinked() -> None:
    # head overlaps body by only 50% of its own area (IoS 0.5 < 0.75).
    detections = [
        _det("page.png", "body", (0, 0, 80, 20), 0.9),
        _det("page.png", "head", (20, 0, 60, 40), 0.8),
    ]

    instances = link_part_detections(detections, iou_containment=0.75)

    assert len(instances) == 2
    assert set(instances[0].parts) == {"body"}
    assert set(instances[1].parts) == {"head"}


def test_face_linked_to_head_without_body() -> None:
    detections = [
        _det("page.png", "head", (30, 20, 70, 120), 0.8),
        _det("page.png", "face", (40, 40, 60, 80), 0.7),
    ]

    instances = link_part_detections(detections)

    assert len(instances) == 1
    assert set(instances[0].parts) == {"head", "face"}


def test_two_heads_in_one_body_expand_to_two_instances() -> None:
    detections = [
        _det("page.png", "body", (0, 0, 300, 200), 0.9),
        _det("page.png", "head", (10, 0, 90, 100), 0.8),
        _det("page.png", "head", (200, 0, 280, 100), 0.7),
    ]

    instances = link_part_detections(detections)

    assert len(instances) == 2
    assert all(set(i.parts) == {"body", "head"} for i in instances)
    assert instances[0].parts["body"]["bbox"] == instances[1].parts["body"]["bbox"]


def test_empty_detections_produce_no_instances() -> None:
    assert link_part_detections([]) == []


def test_crop_path_is_optional_none() -> None:
    assert link_part_detections([_det("p.png", "face", (0, 0, 10, 10), 1.0, None)])


@pytest.mark.parametrize(
    "detection,match",
    [
        (
            {"image": "p.png", "crop_path": None, "bbox": [0, 0, 10, 10], "class_name": "tail", "score": 0.5},
            "class_name must be one of",
        ),
        (
            {"image": "p.png", "crop_path": None, "bbox": [10, 0, 5, 10], "class_name": "face", "score": 0.5},
            "x1 < x2",
        ),
        (
            {"image": "p.png", "crop_path": None, "bbox": [0, 0, 10, 10], "class_name": "face", "score": 1.5},
            "score must lie in",
        ),
        (
            {"image": "p.png", "bbox": [0, 0, 10, 10], "class_name": "face", "score": 0.5},
            "missing keys",
        ),
        (
            {"image": "", "crop_path": None, "bbox": [0, 0, 10, 10], "class_name": "face", "score": 0.5},
            "image must be a non-empty string",
        ),
    ],
)
def test_malformed_detections_fail_loud(detection, match) -> None:
    with pytest.raises(ValueError, match=match):
        link_part_detections([detection])
