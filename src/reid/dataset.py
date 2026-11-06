"""Character-instance crop linking: groups nested face/head/body detections.

Implements the multi-granularity crop-linking layer of plan §5.1: on each
image, Stage 1 detections are grouped into ``CharacterInstance`` records by
hierarchical containment (face ⊂ head ⊂ body via IoS), so per-part embeddings
can later be fused into a single instance descriptor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

# Anatomy hierarchy roots: bodies absorb heads, heads absorb faces.
_PART_CLASS_NAMES = ("body", "head", "face")

_REQUIRED_DETECTION_KEYS = ("image", "crop_path", "bbox", "class_name", "score")


@dataclass
class CharacterInstance:
    """One linked character instance found in a single image.

    Attributes:
        instance_id: stable per-image identifier ("<image>#<001-based idx>").
        image_path: source image the detections were made on.
        parts: present parts only — {"face"|"head"|"body": {"bbox":
            [x1, y1, x2, y2] pixel coords, "score": detector confidence}}.
    """

    instance_id: str
    image_path: str
    parts: dict


def _parse_detection(entry: Any, index: int) -> dict:
    """Validate one raw detection dict into trusted typed state (Law 2).

    Raise:
        ValueError: on malformed entries, naming the offending field.
    """
    if not isinstance(entry, Mapping):
        raise ValueError(
            f"detection[{index}] must be a mapping, got {type(entry).__name__}"
        )

    missing = tuple(key for key in _REQUIRED_DETECTION_KEYS if key not in entry)
    if missing:
        raise ValueError(f"detection[{index}] is missing keys: {missing}")

    image = entry["image"]
    if not isinstance(image, str) or not image:
        raise ValueError(
            f"detection[{index}].image must be a non-empty string, got {image!r}"
        )

    class_name = entry["class_name"]
    if class_name not in _PART_CLASS_NAMES:
        raise ValueError(
            f"detection[{index}].class_name must be one of "
            f"{_PART_CLASS_NAMES}, got {class_name!r}"
        )

    bbox = entry["bbox"]
    if len(bbox) != 4:
        raise ValueError(f"detection[{index}].bbox must have 4 entries, got {bbox!r}")
    x1, y1, x2, y2 = (float(coord) for coord in bbox)
    if not (x1 < x2 and y1 < y2):
        raise ValueError(
            f"detection[{index}].bbox must satisfy x1 < x2 and y1 < y2, got {bbox!r}"
        )

    score = entry["score"]
    score = float(score)
    if not 0.0 <= score <= 1.0:
        raise ValueError(
            f"detection[{index}].score must lie in [0.0, 1.0], got {score}"
        )

    return {
        "image": image,
        "crop_path": entry["crop_path"],
        "bbox": (x1, y1, x2, y2),
        "class_name": class_name,
        "score": score,
    }


def _ios_of_child_in_parent(child: tuple[float, ...], parent: tuple[float, ...]) -> float:
    """Intersection-over-Self: |child ∩ parent| / |child|.

    child area is validated non-degenerate by ``_parse_detection``.
    """
    ix1, iy1 = max(child[0], parent[0]), max(child[1], parent[1])
    ix2, iy2 = min(child[2], parent[2]), min(child[3], parent[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    child_area = (child[2] - child[0]) * (child[3] - child[1])
    return intersection / child_area


def _best_containing_parent(
    child_bbox: tuple[float, ...],
    parents: list[dict],
    iou_containment: float,
) -> int | None:
    """Index of the parent containing ``child_bbox`` with the highest IoS.

    ``parents`` must already be in deterministic (bbox, score) order, so an
    exact IoS tie resolves to the earliest parent.
    """
    best_index: int | None = None
    best_ios = 0.0
    for index, parent in enumerate(parents):
        ios = _ios_of_child_in_parent(child_bbox, parent["bbox"])
        if ios > best_ios:
            best_ios = ios
            best_index = index

    if best_index is None or best_ios < iou_containment:
        return None
    return best_index


def _part_entry(detection: dict) -> dict:
    """The per-part manifest fragment nested inside a CharacterInstance."""
    return {
        "bbox": [detection["bbox"][0], detection["bbox"][1], detection["bbox"][2], detection["bbox"][3]],
        "score": detection["score"],
    }


def link_part_detections(
    detections: list[dict], iou_containment: float = 0.75
) -> list[CharacterInstance]:
    """Group face/head/body detections into per-image character instances.

    Linking follows the anatomy hierarchy face ⊂ head ⊂ body on each image
    independently: a face joins its best containing head (highest IoS, ties
    resolved by pre-sorted order), that head joins its best containing body,
    and any detection left without a qualifying parent becomes the root of
    its own single-part instance.  ``parts`` holds at most one box per class,
    so a chain that branches into several same-class children (e.g. two heads
    inside one wide body box) expands into one instance per leaf, reusing the
    shared ancestors.  Output ordering is fully deterministic: images sorted
    lexicographically; within an image, body-rooted instances first (in body
    (bbox, score) order, then head order, then face order), then orphan heads
    and finally orphan faces, each in (bbox, score) order.

    Args:
        detections: entries with keys image, crop_path (str | None), bbox
            (pixel [x1, y1, x2, y2]), class_name ("body"|"head"|"face"),
            score (float in [0, 1]).
        iou_containment: minimum IoS of a child inside its parent to link.

    Raise:
        ValueError: on any malformed detection entry (see ``_parse_detection``).
    """
    # --- Law 2: parse everything into trusted state before any linking
    parsed = [_parse_detection(entry, index) for index, entry in enumerate(detections)]

    instances: list[CharacterInstance] = []
    for image in sorted({det["image"] for det in parsed}):
        image_dets = [det for det in parsed if det["image"] == image]
        # Deterministic processing order: sort by bbox then score.
        ordered = sorted(image_dets, key=lambda det: (det["bbox"], det["score"]))
        bodies = [det for det in ordered if det["class_name"] == "body"]
        heads = [det for det in ordered if det["class_name"] == "head"]
        faces = [det for det in ordered if det["class_name"] == "face"]

        # Best containing body per head, best containing head per face.
        head_to_body = [
            _best_containing_parent(head["bbox"], bodies, iou_containment)
            for head in heads
        ]
        face_to_head = [
            _best_containing_parent(face["bbox"], heads, iou_containment)
            for face in faces
        ]

        heads_by_body: dict[int, list[int]] = {}
        for head_index, body_index in enumerate(head_to_body):
            heads_by_body.setdefault(body_index, []).append(head_index)
        faces_by_head: dict[int, list[int]] = {}
        for face_index, head_index in enumerate(face_to_head):
            if head_index is not None:
                faces_by_head.setdefault(head_index, []).append(face_index)

        instance_counter = 0

        # Instance assembly — one CharacterInstance per maximal F⊂H⊂B chain.
        # ``parts`` holds at most one box per class, so a chain branching into
        # several same-class children (two heads inside one wide body box)
        # expands into one instance per leaf, reusing the shared ancestors.
        def _emit(parts: dict) -> None:
            nonlocal instance_counter
            instance_counter += 1
            instances.append(_build_instance(image, instance_counter, parts))

        for body_index, body in enumerate(bodies):
            body_parts = {"body": _part_entry(body)}
            body_heads = heads_by_body.get(body_index, [])
            if not body_heads:
                _emit(dict(body_parts))
                continue
            for head_index in body_heads:
                head_parts = {**body_parts, "head": _part_entry(heads[head_index])}
                head_faces = faces_by_head.get(head_index, [])
                if not head_faces:
                    _emit(head_parts)
                    continue
                for face_index in head_faces:
                    _emit({**head_parts, "face": _part_entry(faces[face_index])})

        for head_index, head in enumerate(heads):
            if head_to_body[head_index] is not None:
                continue  # already covered by a body-rooted chain
            head_parts = {"head": _part_entry(head)}
            head_faces = faces_by_head.get(head_index, [])
            if not head_faces:
                _emit(head_parts)
                continue
            for face_index in head_faces:
                _emit({**head_parts, "face": _part_entry(faces[face_index])})

        for face_index, face in enumerate(faces):
            if face_to_head[face_index] is not None:
                continue  # already covered by a head-rooted chain
            _emit({"face": _part_entry(face)})

    return instances


def _build_instance(image: str, index: int, parts: dict) -> CharacterInstance:
    """Assemble one CharacterInstance with its stable per-image id."""
    return CharacterInstance(
        instance_id=f"{image}#{index:03d}",
        image_path=image,
        parts=parts,
    )
