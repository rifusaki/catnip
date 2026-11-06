#!/usr/bin/env python3
"""pseudo_label.py — Cross-dataset synthetic pseudo-labeling with geometric verification.

Manga109 staging only annotates ``0: body`` + ``2: face`` while AnimeHeads
staging (``ah_yolo``, ``ah_coco``) only annotates ``1: head``. Joint training
on those partial labels pushes negative BCE gradients for the missing classes
and teaches a domain shortcut. This script fills the gaps using external
teacher detectors (run on Colab L4) whose predictions are accepted only when
they pass **hard geometric verification** against nested existing GT anatomy
(``face ⊂ head ⊂ body``), plus a 4-parameter face box adapter
(``src/preprocess/box_calibration.py``) mapping loose Danbooru face boxes onto
the tight Manga109 convention.

DESTRUCTIVE-DATA POLICY (non-negotiable):
    The script NEVER mutates staging inputs. It writes an ENRICHED OUTPUT
    staging tree: images as symlinks, labels regenerated from the pristine
    originals plus the synthetic lines. Reruns are idempotent (byte-identical
    output). AnimeHeads headshot-only crops (no verified torso) are
    downsampled to 10% deterministically; the images dropped by that draw are
    EXCLUDED from the output tree entirely (no symlink, no label) so unified
    training stats stay honest.

Usage::

    python scripts/unify/pseudo_label.py --target-section all \
        --adapter config/face_box_adapter.json \
        --catnip-model models/yolo26_stage1_body_head_face.pt \
        --device cuda

    python scripts/unify/pseudo_label.py --target-section manga109 --dry-run
    python scripts/unify/pseudo_label.py --target-section anime --limit 20
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

# Allow running the script directly from scripts/unify/ without an installed package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from src.eval.adapters import Detection, get_detector
from src.preprocess.box_calibration import FaceBoxAdapter, FaceBoxAdapterError

logger = logging.getLogger("pseudo_label")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_BODY = 0
CLASS_HEAD = 1
CLASS_FACE = 2
CLASS_NAMES = {CLASS_BODY: "body", CLASS_HEAD: "head", CLASS_FACE: "face"}

IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})

# Processing regimes: Manga109 gets verified head labels, AnimeHeads gets
# calibrated faces and verified bodies plus headshot filtering.
REGIME_MANGA109 = "manga109"
REGIME_ANIME = "anime"
# Calibration runs face teachers over the GT-face-bearing manga sources
# (plan task 3.2 step 1: izutsumi + a sample of manga109).
REGIME_CALIBRATE = "calibrate"
ANIME_SOURCE_NAMES = ("ah_yolo", "ah_coco")
MANGA109_SOURCE_NAME = "manga109"
CALIBRATE_SOURCE_NAMES = ("izutsumi", MANGA109_SOURCE_NAME)
SECTION_SOURCES: dict[str, tuple[tuple[str, str], ...]] = {
    "manga109": ((MANGA109_SOURCE_NAME, REGIME_MANGA109),),
    "anime": tuple((name, REGIME_ANIME) for name in ANIME_SOURCE_NAMES),
    "all": (
        (MANGA109_SOURCE_NAME, REGIME_MANGA109),
        *tuple((name, REGIME_ANIME) for name in ANIME_SOURCE_NAMES),
    ),
    "calibrate": tuple((name, REGIME_CALIBRATE) for name in CALIBRATE_SOURCE_NAMES),
}

# Teacher inference confidences (plan section 3, steps 3.2 & 3.3).
CONF_HEAD_TEACHER = 0.45
CONF_BACK_HEAD = 0.65      # Rule B back-of-head confidence floor
CONF_FACE_TEACHER = 0.45
CONF_BODY_TEACHER = 0.50

# Geometric verification thresholds.
WBF_MATCH_IOS = 0.50
RULE_A_FACE_IOS = 0.80
RULE_A_AREA_MIN = 1.15
RULE_A_AREA_MAX = 4.50
RULE_B_HEAD_IN_TOP_BAND = 0.35
RULE_B_HEAD_IN_BODY_IOS = 0.75
FACE_ASSOCIATED_WITH_BODY_IOS = 0.50

# Headshot-only AnimeHeads crops: keep a deterministic 10%.
DOWNSAMPLE_KEEP_PROB = 0.10
DEFAULT_SEED = 42


# ========================================================================
# Pure box geometry on normalized xyxy Detection boxes
# ========================================================================

def _box_area(box: Detection) -> float:
    return max((box.x2 - box.x1) * (box.y2 - box.y1), 0.0)


def _intersection_area(a: Detection, b: Detection) -> float:
    inter_w = min(a.x2, b.x2) - max(a.x1, b.x1)
    inter_h = min(a.y2, b.y2) - max(a.y1, b.y1)
    if inter_w <= 0.0 or inter_h <= 0.0:
        return 0.0
    return inter_w * inter_h


def box_iou(a: Detection, b: Detection) -> float:
    """Intersection over union of two normalized xyxy boxes."""
    inter = _intersection_area(a, b)
    union = _box_area(a) + _box_area(b) - inter
    return inter / union if union > 0.0 else 0.0


def box_ios(subject: Detection, container: Detection) -> float:
    """Intersection over the *subject* area: fraction of *subject* inside *container*."""
    subjects_area = _box_area(subject)
    if subjects_area <= 0.0:
        return 0.0
    return _intersection_area(subject, container) / subjects_area


def box_ios_smallest(a: Detection, b: Detection) -> float:
    """Intersection over the smaller area (containment-tolerant fusion metric)."""
    smallest = min(_box_area(a), _box_area(b))
    if smallest <= 0.0:
        return 0.0
    return _intersection_area(a, b) / smallest


def to_cxcywh(box: Detection) -> tuple[float, float, float, float]:
    """Convert a normalized xyxy detection to ``(xc, yc, w, h)``."""
    return (
        (box.x1 + box.x2) / 2.0,
        (box.y1 + box.y2) / 2.0,
        box.x2 - box.x1,
        box.y2 - box.y1,
    )


def from_cxcywh(xc: float, yc: float, w: float, h: float, score: float, class_id: int) -> Detection:
    """Build a normalized xyxy detection from ``(xc, yc, w, h)`` values."""
    return Detection(
        x1=xc - w / 2.0,
        y1=yc - h / 2.0,
        x2=xc + w / 2.0,
        y2=yc + h / 2.0,
        score=score,
        class_id=class_id,
    )


def clamp_to_cover(outer: Detection, inner: Detection) -> Detection:
    """Expand *outer* minimally so it fully covers *inner*."""
    return Detection(
        x1=min(outer.x1, inner.x1),
        y1=min(outer.y1, inner.y1),
        x2=max(outer.x2, inner.x2),
        y2=max(outer.y2, inner.y2),
        score=outer.score,
        class_id=outer.class_id,
    )


# ========================================================================
# Weighted Box Fusion (local, pure)
# ========================================================================

def weighted_box_fusion(detections: list[Detection], match_ios: float = WBF_MATCH_IOS) -> list[Detection]:
    """Fuse overlapping teacher predictions into consensus boxes.

    Greedy clustering in score-descending order: a detection joins the first
    cluster whose running fused box overlaps it with IoS >= *match_ios*
    (intersection over the smaller area, so nested boxes fuse too). Fused
    coordinates are score-weighted averages; the fused confidence is the
    member mean. Deterministic for a given input list.
    """
    if not detections:
        return []

    ordered = sorted(range(len(detections)), key=lambda i: (-detections[i].score, i))
    clusters: list[list[Detection]] = []
    fused: list[Detection] = []

    for index in ordered:
        detection = detections[index]
        for cluster_index, fused_box in enumerate(fused):
            if box_ios_smallest(detection, fused_box) >= match_ios:
                clusters[cluster_index].append(detection)
                fused[cluster_index] = _fuse_cluster(clusters[cluster_index])
                break
        else:
            clusters.append([detection])
            fused.append(detection)

    return sorted(fused, key=lambda box: (-box.score, box.x1, box.y1))


def _fuse_cluster(members: list[Detection]) -> Detection:
    """Score-weighted average of cluster coordinates; fused score is the mean."""
    weights = [member.score for member in members]
    total_weight = sum(weights)
    if total_weight <= 0.0:
        weights = [1.0] * len(members)
        total_weight = float(len(members))

    x1 = sum(w * m.x1 for w, m in zip(weights, members)) / total_weight
    y1 = sum(w * m.y1 for w, m in zip(weights, members)) / total_weight
    x2 = sum(w * m.x2 for w, m in zip(weights, members)) / total_weight
    y2 = sum(w * m.y2 for w, m in zip(weights, members)) / total_weight

    return Detection(
        x1=x1,
        y1=y1,
        x2=max(x1, x2),
        y2=max(y1, y2),
        score=min(1.0, total_weight / len(members)),
        class_id=members[0].class_id,
    )


# ========================================================================
# Geometric verifiers (pure — no I/O, unit-tested with synthetic boxes)
# ========================================================================

def verify_manga109_heads(
    teacher_heads: list[Detection],
    catnip_heads: list[Detection],
    gt_faces: list[Detection],
    gt_bodies: list[Detection],
    conf_head: float = CONF_HEAD_TEACHER,
    conf_back: float = CONF_BACK_HEAD,
) -> list[Detection]:
    """Accept fused head candidates verified against Manga109 GT anatomy.

    The two teacher pools are WBF-fused first (IoS >= 0.5). A candidate head
    ``H`` is accepted iff either rule passes:

    Rule A (face enclosure): ``H`` encloses some GT face ``F`` with
    ``IoS(F, H) >= 0.80`` (the pre-clamp "encloses" test — disjoint boxes can
    never qualify), then ``H`` is clamped outward to fully cover ``F`` *before*
    the area-ratio check, so the final box covering ratio satisfies
    ``1.15 <= area(H)/area(F) <= 4.5``. The accepted box is the clamped one.

    Rule B (back-of-head): for a GT body ``B`` with no associated GT face
    (no face with ``IoS(face, B) >= 0.5``), ``H`` has confidence >=
    *conf_back*, the whole head box lies within the top 35% band of ``B``
    (all four edges contained in the band region), and ``IoS(H, B) >= 0.75``.
    """
    candidates = weighted_box_fusion(list(teacher_heads) + list(catnip_heads))
    bodies_without_faces = [
        body for body in gt_bodies
        if not any(box_ios(face, body) >= FACE_ASSOCIATED_WITH_BODY_IOS for face in gt_faces)
    ]

    accepted: list[Detection] = []
    for candidate in candidates:
        verified = _verify_head_rule_a(candidate, gt_faces, conf_head)
        if verified is None:
            verified = _verify_head_rule_b(candidate, bodies_without_faces, conf_back)
        if verified is not None:
            accepted.append(verified)

    return accepted


def _verify_head_rule_a(candidate: Detection, gt_faces: list[Detection], conf_head: float) -> Detection | None:
    """Face-enclosure rule: returns the clamped head box when a GT face verifies it."""
    if candidate.score < conf_head:
        return None
    for face in gt_faces:
        if box_ios(face, candidate) < RULE_A_FACE_IOS:
            continue
        covering_head = clamp_to_cover(candidate, face)
        area_ratio = _box_area(covering_head) / max(_box_area(face), 1e-12)
        if RULE_A_AREA_MIN <= area_ratio <= RULE_A_AREA_MAX:
            return replace(covering_head, class_id=CLASS_HEAD)
    return None


def _verify_head_rule_b(candidate: Detection, bodies_without_faces: list[Detection], conf_back: float) -> Detection | None:
    """Back-of-head rule: high-confidence heads atop faceless bodies.

    Plan letter: ``H`` LIES WITHIN the top 35% band of ``B`` — the whole
    head box is contained in the band region (all four edges), not merely
    its center — with confidence >= *conf_back* and ``IoS(H, B) >= 0.75``.
    """
    if candidate.score < conf_back:
        return None
    for body in bodies_without_faces:
        band_bottom = body.y1 + RULE_B_HEAD_IN_TOP_BAND * (body.y2 - body.y1)
        within_top_band = (
            candidate.y1 >= body.y1 - 1e-6
            and candidate.y2 <= band_bottom + 1e-6
            and candidate.x1 >= body.x1 - 1e-6
            and candidate.x2 <= body.x2 + 1e-6
        )
        if within_top_band and box_ios(candidate, body) >= RULE_B_HEAD_IN_BODY_IOS:
            return replace(candidate, class_id=CLASS_HEAD)
    return None


def verify_anime_faces(
    teacher_faces: list[Detection],
    gt_heads: list[Detection],
    adapter: FaceBoxAdapter,
    conf_face: float = CONF_FACE_TEACHER,
    ios_in_head: float = 0.85,
    max_per_head: int = 1,
) -> list[Detection]:
    """Accept loose face predictions strictly inside GT heads, adapter-transformed.

    A face prediction qualifies only when ``IoS(pred, gt_head) >= ios_in_head``
    (fraction of the prediction inside the GT head) and its confidence clears
    *conf_face*. At most *max_per_head* predictions are kept per GT head, the
    highest-scored first. Accepted faces are shrunk onto the tight Manga109
    convention via *adapter* and returned as class ``2`` detections.
    """
    accepted: list[Detection] = []
    for head in gt_heads:
        inside = [
            face for face in teacher_faces
            if face.score >= conf_face and box_ios(face, head) >= ios_in_head
        ]
        inside.sort(key=lambda face: (-face.score, face.x1, face.y1))
        for face in inside[:max_per_head]:
            accepted.append(_adapt_face_to_manga_convention(face, adapter))
    return accepted


def _adapt_face_to_manga_convention(face: Detection, adapter: FaceBoxAdapter) -> Detection:
    xc, yc, w, h = to_cxcywh(face)
    transformed = adapter.transform((xc, yc, w, h))
    return from_cxcywh(*transformed, score=face.score, class_id=CLASS_FACE)


def verify_anime_bodies(
    teacher_bodies: list[Detection],
    gt_heads: list[Detection],
    conf_body: float = CONF_BODY_TEACHER,
    ios_head_in_body: float = 0.70,
    iou_torso: float = 0.80,
) -> list[Detection]:
    """Accept body predictions that enclose a GT head AND include a torso.

    A body qualifies when it encloses at least one GT head with
    ``IoS(head, body) >= 0.70`` and is genuinely larger than the head
    (``IoU(body, head) <= 0.80``) — a person crop, not a headshot rectangle.
    """
    accepted: list[Detection] = []
    for body in teacher_bodies:
        if body.score < conf_body:
            continue
        has_verified_head = any(
            box_ios(head, body) >= ios_head_in_body and box_iou(body, head) <= iou_torso
            for head in gt_heads
        )
        if has_verified_head:
            accepted.append(replace(body, class_id=CLASS_BODY))
    return accepted


def downsample_headshots(
    image_names: list[str],
    valid_body_counts: dict[str, int],
    keep_prob: float = DOWNSAMPLE_KEEP_PROB,
    seed: int = DEFAULT_SEED,
) -> set[str]:
    """Deterministically decide which headshot-only images survive.

    Images with at least one valid body always survive. For images with zero
    bodies each draw uses its own per-filename RNG seeded from ``(seed,
    filename)`` — so results never change across reruns, re-orderings, or
    parallel shards.

    Returns the set of kept image names.
    """
    if not 0.0 <= keep_prob <= 1.0:
        raise ValueError(f"keep_prob must be within [0.0, 1.0], got {keep_prob}")

    kept: set[str] = set()
    for name in image_names:
        if valid_body_counts.get(name, 0) >= 1:
            kept.add(name)
            continue
        draw = random.Random(f"{seed}:{name}").random()
        if draw < keep_prob:
            kept.add(name)
    return kept


# ========================================================================
# Staging label parsing  (Law 2 — parse at the boundary)
# ========================================================================

def read_staging_label(label_path: Path) -> tuple[list[str], dict[int, list[Detection]]]:
    """Read a pristine YOLO staging label.

    Returns:
        verbatim non-empty lines, and per-class normalized xyxy Detection
        boxes (score 1.0). Malformed lines are skipped with a warning.
    """
    lines = [
        raw for raw in label_path.read_text(encoding="utf-8").splitlines() if raw.strip()
    ]
    boxes_by_class: dict[int, list[Detection]] = {}
    for raw in lines:
        parts = raw.split()
        if len(parts) != 5:
            logger.warning("[%s] Malformed label line skipped: %r", label_path.name, raw)
            continue
        try:
            class_id = int(parts[0])
            xc, yc, w, h = (float(part) for part in parts[1:5])
        except ValueError:
            logger.warning("[%s] Non-numeric label line skipped: %r", label_path.name, raw)
            continue
        boxes_by_class.setdefault(class_id, []).append(
            from_cxcywh(xc, yc, w, h, score=1.0, class_id=class_id)
        )
    return lines, boxes_by_class


def format_yolo_line(class_id: int, box: Detection, precision: int = 6) -> str:
    """Format a normalized xyxy detection as a YOLO center-format label line."""
    xc, yc, w, h = to_cxcywh(box)
    return f"{class_id} {xc:.{precision}f} {yc:.{precision}f} {w:.{precision}f} {h:.{precision}f}"


def find_image_for_label(stem: str, images_dir: Path) -> Path | None:
    """Return the image file matching *stem* in *images_dir*, or None."""
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


# ========================================================================
# Teacher inference seam (monkeypatched in tests; real path needs weights)
# ========================================================================

def run_teachers(
    image_path: Path,
    model_key: str,
    target: str,
    conf: float,
    device: str = "cpu",
    custom_path: Path | None = None,
) -> list[Detection]:
    """Run one catalog teacher detector over a staging image.

    This is the single I/O + inference seam of the pipeline: everything else
    in this module is pure geometry and file orchestration, so tests craft
    teacher outputs by monkeypatching this function.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Cannot read staging image for teacher inference: {image_path}")
    detector = get_detector(model_key, target=target, custom_path=custom_path, device=device)
    return detector.predict(image, conf=conf)


# ========================================================================
# Per-image enrichment
# ========================================================================

@dataclass
class ImageRecord:
    """Everything the writer and the manifest need for one staging image."""

    image_path: Path | None
    label_path: Path
    image_name: str
    original_lines: list[str] = field(default_factory=list)
    synthetic_lines: list[str] = field(default_factory=list)
    kept: bool = False
    reason: str = "pending"
    added: dict[str, int] = field(default_factory=dict)
    bodies_found: int = 0
    downsampled: bool = False

    @property
    def label_name(self) -> str:
        return self.label_path.name

    def manifest_entry(self) -> dict:
        return {
            "kept": self.kept,
            "reason": self.reason,
            "added": dict(sorted(self.added.items())),
            "bodies_found": self.bodies_found,
            "downsampled": self.downsampled,
        }


def enrich_manga109_image(
    image_path: Path,
    label_path: Path,
    *,
    device: str,
    catnip_model: Path,
) -> ImageRecord:
    """Generate verified ``1: head`` labels for one Manga109 staging image."""
    lines, classes = read_staging_label(label_path)
    gt_faces = classes.get(CLASS_FACE, [])
    gt_bodies = classes.get(CLASS_BODY, [])

    teacher_heads = run_teachers(image_path, "deepghs_head_x", "head", CONF_HEAD_TEACHER, device)
    catnip_heads = run_teachers(
        image_path, "catnip", "head", CONF_HEAD_TEACHER, device, custom_path=catnip_model
    )
    accepted_heads = verify_manga109_heads(teacher_heads, catnip_heads, gt_faces, gt_bodies)

    synthetic_lines = [format_yolo_line(CLASS_HEAD, head) for head in accepted_heads]
    return ImageRecord(
        image_path=image_path,
        label_path=label_path,
        image_name=image_path.name,
        original_lines=lines,
        synthetic_lines=synthetic_lines,
        kept=True,
        reason="enriched",
        added={str(CLASS_HEAD): len(accepted_heads)},
        bodies_found=len(gt_bodies),
    )


def enrich_anime_image(
    image_path: Path,
    label_path: Path,
    *,
    device: str,
    adapter: FaceBoxAdapter,
) -> ImageRecord:
    """Generate calibrated ``2: face`` and verified ``0: body`` labels for one AnimeHeads image."""
    lines, classes = read_staging_label(label_path)
    gt_heads = classes.get(CLASS_HEAD, [])

    deepghs_faces = run_teachers(image_path, "deepghs_face_s", "face", CONF_FACE_TEACHER, device)
    zymk9_faces = run_teachers(image_path, "zymk9_face", "face", CONF_FACE_TEACHER, device)
    fused_faces = weighted_box_fusion(list(deepghs_faces) + list(zymk9_faces))
    accepted_faces = verify_anime_faces(fused_faces, gt_heads, adapter)

    teacher_bodies = run_teachers(image_path, "deepghs_person_s", "body", CONF_BODY_TEACHER, device)
    accepted_bodies = verify_anime_bodies(teacher_bodies, gt_heads)

    synthetic_lines = [format_yolo_line(CLASS_FACE, face) for face in accepted_faces]
    synthetic_lines += [format_yolo_line(CLASS_BODY, body) for body in accepted_bodies]
    return ImageRecord(
        image_path=image_path,
        label_path=label_path,
        image_name=image_path.name,
        original_lines=lines,
        synthetic_lines=synthetic_lines,
        added={str(CLASS_FACE): len(accepted_faces), str(CLASS_BODY): len(accepted_bodies)},
        bodies_found=len(accepted_bodies),
    )


# ========================================================================
# Face box adapter calibration (plan task 3.2)
# ========================================================================

CALIBRATION_MATCH_IOU = 0.45
CALIBRATION_VALIDATE_IOU = 0.75
CALIBRATION_MIN_QUALITY = 0.45
CALIBRATION_FIT_FRACTION = 0.8
# Plan task 3.2 step 1: "a 2,000-page sample of data/staging/manga109".
# The sample is STRIDED across the sorted label listing (a plain head-cap
# would sample only the first couple of titles).
CALIBRATION_MANGA109_SAMPLE = 2_000


def _strided_sample(records: list[ImageRecord], sample: int) -> list[ImageRecord]:
    """Deterministic strided subsample of sorted records (empty records kept).

    Coverage can come out slightly under *sample* when the page count sits
    between one and two multiples of it (ceil stride) — acceptable for a
    page-level sanity sample.
    """
    if sample <= 0 or len(records) <= sample:
        return records
    stride = -(-len(records) // sample)  # ceil division, sample coverage >= sample
    return records[::stride][:sample]


def calibrate_face_adapter(
    staging_root: Path,
    output_path: Path,
    *,
    device: str,
    limit: int,
    dry_run: bool,
    match_iou: float = CALIBRATION_MATCH_IOU,
    validate_iou: float = CALIBRATION_VALIDATE_IOU,
    min_quality: float = CALIBRATION_MIN_QUALITY,
    fit_fraction: float = CALIBRATION_FIT_FRACTION,
    manga109_sample: int = CALIBRATION_MANGA109_SAMPLE,
) -> dict:
    """Fit the face box convention adapter on GT-face-bearing manga sources.

    Plan task 3.2: run the face teachers (deepghs_face_s WBF-fused with
    zymk9_face, conf >= 0.45) over the izutsumi and manga109 staging dirs,
    pair each prediction with its ground-truth tight face PER IMAGE at
    IoU >= *match_iou* (pool matching across images would scramble pairs),
    split each source's pairs deterministically (sorted image order; the
    first *fit_fraction* per source trains the fit, the rest validate — so
    both sources appear in the validation set), Huber-fit the 4-parameter
    adapter on the aligned pairs, and assert the aligned validation split
    clears the quality floor before serializing to JSON.

    *limit* caps the manga109 sample with a strided draw. In calibrate mode
    ``limit=0`` keeps the plan's 2,000-page strided default for manga109;
    to scan EVERY manga109 page pass a value at least as large as the page
    count.

    Returns the fit/validation report. Raises FaceBoxAdapterError if fewer
    than the minimum pairs match or the fit degenerates; failure of the
    validation floor raises FaceBoxAdapterError from validate_adapter_from_pairs.
    """
    from src.preprocess.box_calibration import (
        fit_face_box_adapter_from_pairs,
        match_pairs,
        validate_adapter_from_pairs,
    )

    fit_pairs_by_source: dict[str, list] = {}
    validation_pairs_by_source: dict[str, list] = {}
    images_scanned = 0
    for source_name in CALIBRATE_SOURCE_NAMES:
        staging_dir = staging_root / source_name
        if not staging_dir.is_dir():
            raise FileNotFoundError(f"Calibration staging source not found: {staging_dir}")
        sample = limit if (limit > 0 and source_name == MANGA109_SOURCE_NAME) else manga109_sample
        records = collect_staging_pairs(staging_dir, 0)
        if source_name == MANGA109_SOURCE_NAME:
            records = _strided_sample(records, sample)

        source_pairs: list[tuple[tuple[float, float, float, float], tuple[float, float, float, float]]] = []
        for record in records:
            if record.image_path is None:
                continue
            images_scanned += 1
            _, classes = read_staging_label(record.label_path)
            gt_faces = classes.get(CLASS_FACE, [])
            if not gt_faces:
                continue
            deepghs_faces = run_teachers(
                record.image_path, "deepghs_face_s", "face", CONF_FACE_TEACHER, device
            )
            zymk9_faces = run_teachers(
                record.image_path, "zymk9_face", "face", CONF_FACE_TEACHER, device
            )
            fused_faces = weighted_box_fusion(list(deepghs_faces) + list(zymk9_faces))
            if not fused_faces:
                continue
            preds = np.asarray([to_cxcywh(face) for face in fused_faces], dtype=np.float64)
            gts = np.asarray([to_cxcywh(face) for face in gt_faces], dtype=np.float64)
            for pred_idx, gt_idx, _iou in match_pairs(preds, gts, match_iou):
                source_pairs.append((tuple(preds[pred_idx]), tuple(gts[gt_idx])))

        fit_count = max(1, int(len(source_pairs) * fit_fraction))
        fit_pairs_by_source[source_name] = source_pairs[:fit_count]
        validation_pairs_by_source[source_name] = source_pairs[fit_count:]

    all_fit_pairs = [
        pair for source_name in CALIBRATE_SOURCE_NAMES for pair in fit_pairs_by_source[source_name]
    ]
    all_validation_pairs = [
        pair for source_name in CALIBRATE_SOURCE_NAMES for pair in validation_pairs_by_source[source_name]
    ]
    if not all_fit_pairs:
        raise FaceBoxAdapterError(
            "No (prediction, ground truth) face pairs matched — cannot fit the "
            "adapter. Check that the staging dirs carry class-2 GT faces."
        )

    fit_predictions = np.asarray([pair[0] for pair in all_fit_pairs], dtype=np.float64)
    fit_ground_truths = np.asarray([pair[1] for pair in all_fit_pairs], dtype=np.float64)
    adapter = fit_face_box_adapter_from_pairs(fit_predictions, fit_ground_truths)

    report = {
        "images_scanned": images_scanned,
        "pairs_matched": len(all_fit_pairs) + len(all_validation_pairs),
        "fit_pairs_by_source": {
            name: len(fit_pairs_by_source[name]) for name in CALIBRATE_SOURCE_NAMES
        },
        "validation_pairs_by_source": {
            name: len(validation_pairs_by_source[name]) for name in CALIBRATE_SOURCE_NAMES
        },
    }
    if all_validation_pairs:
        validation_predictions = np.asarray([pair[0] for pair in all_validation_pairs], dtype=np.float64)
        validation_ground_truths = np.asarray(
            [pair[1] for pair in all_validation_pairs], dtype=np.float64
        )
        validation = validate_adapter_from_pairs(
            adapter,
            validation_predictions,
            validation_ground_truths,
            iou_threshold=validate_iou,
            min_quality=min_quality,
        )
        report["validation"] = validation
        report["validate_iou"] = validate_iou

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        adapter.to_json(output_path)
    report["adapter_path"] = str(output_path) if not dry_run else None
    return report


def _print_calibration_report(report: dict, dry_run: bool) -> None:
    mode = "DRY RUN — adapter NOT written" if dry_run else "COMPLETE"
    print(f"\n{'=' * 60}\n{' Face Box Adapter Calibration — ' + mode:=^60}\n{'=' * 60}")
    print(f"  images scanned:      {report['images_scanned']:>6}")
    print(f"  face pairs matched:   {report['pairs_matched']:>6}")
    validation = report.get("validation")
    if validation:
        validate_iou = report.get("validate_iou", CALIBRATION_VALIDATE_IOU)
        print(f"  validation pairs:     {validation['matched']:>6}")
        print(
            f"  mean IoU before/after:      "
            f"{validation['mean_iou_before']:.3f} -> {validation['mean_iou_after']:.3f}"
        )
        print(
            f"  recall@{validate_iou:.2f} before/after:  "
            f"{validation['recall_at_iou_before']:.3f} -> "
            f"{validation['recall_at_iou_after']:.3f}"
        )
    if report.get("adapter_path"):
        print(f"  adapter written:      {report['adapter_path']}")
    print(f"{'=' * 60}\n")


# ========================================================================
# Per-source orchestration & writer  (thin; reading pristine, writing enriched)
# ========================================================================

def collect_staging_pairs(staging_dir: Path, limit: int) -> list[ImageRecord]:
    """Pair sorted staging label files with their images (records; kept unset)."""
    images_dir, labels_dir = staging_dir / "images", staging_dir / "labels"
    label_files = sorted(labels_dir.glob("*.txt"))

    records: list[ImageRecord] = []
    unpaired = 0
    for label_path in label_files:
        image_path = find_image_for_label(label_path.stem, images_dir)
        if image_path is None:
            unpaired += 1
            records.append(
                ImageRecord(
                    image_path=None,
                    label_path=label_path,
                    image_name=label_path.stem,
                    reason="missing_image",
                )
            )
            continue
        records.append(
            ImageRecord(image_path=image_path, label_path=label_path, image_name=image_path.name)
        )
    if unpaired:
        logger.warning("[%s] %d label files have no matching image.", staging_dir.name, unpaired)

    return records[:limit] if limit > 0 else records


def process_source(
    source_name: str,
    regime: str,
    staging_root: Path,
    output_root: Path,
    *,
    device: str,
    catnip_model: Path | None,
    adapter: FaceBoxAdapter | None,
    limit: int,
    seed: int,
    dry_run: bool,
) -> dict:
    """Enrich one staging source; write its enriched tree unless *dry_run*."""
    staging_dir = staging_root / source_name
    if not staging_dir.is_dir():
        raise FileNotFoundError(f"Staging directory not found: {staging_dir}")
    if not (staging_dir / "images").is_dir() or not (staging_dir / "labels").is_dir():
        raise FileNotFoundError(f"Staging source lacks images/ or labels/ subdirs: {staging_dir}")
    if regime == REGIME_ANIME and adapter is None:
        raise ValueError("AnimeHeads enrichment requires a face box adapter")

    records = collect_staging_pairs(staging_dir, limit)

    enriched: list[ImageRecord] = []
    for record in records:
        if record.image_path is None:
            enriched.append(record)
            continue
        if regime == REGIME_MANGA109:
            enriched.append(
                enrich_manga109_image(
                    record.image_path, record.label_path, device=device, catnip_model=catnip_model
                )
            )
        else:
            enriched.append(
                enrich_anime_image(
                    record.image_path, record.label_path, device=device, adapter=adapter
                )
            )

    if regime == REGIME_ANIME:
        _apply_headshot_downsampling(enriched, seed)

    totals = summarize_source(source_name, enriched)
    if not dry_run:
        write_source_tree(source_name, output_root, enriched, totals)
    return totals


def _apply_headshot_downsampling(records: list[ImageRecord], seed: int) -> None:
    """Mark zero-body images as kept/excluded by the deterministic 10% draw."""
    body_counts = {record.image_name: record.bodies_found for record in records}
    headshot_names = [record.image_name for record in records if record.bodies_found == 0]
    kept_names = downsample_headshots(headshot_names, body_counts, DOWNSAMPLE_KEEP_PROB, seed)

    for record in records:
        if record.bodies_found >= 1:
            record.kept = True
            record.reason = "enriched"
            continue
        record.downsampled = True
        if record.image_name in kept_names:
            record.kept = True
            record.reason = "headshot_kept"
        else:
            record.kept = False
            record.reason = "headshot_excluded"


def summarize_source(source_name: str, records: list[ImageRecord]) -> dict:
    """Aggregate per-image records into the manifest totals block.

    Only kept images contribute to ``added_boxes``: excluded headshots never
    reach the output tree, so counting their synthetic boxes would overstate
    what a downstream ``unify`` run actually sees.
    """
    added_by_class = {str(class_id): 0 for class_id in CLASS_NAMES}
    for record in records:
        if not record.kept:
            continue
        for class_key, count in record.added.items():
            added_by_class[class_key] = added_by_class.get(class_key, 0) + count

    kept = [record for record in records if record.kept]
    return {
        "source": source_name,
        "images_seen": len(records),
        "images_kept": len(kept),
        "images_excluded": len(records) - len(kept),
        "headshot_only": sum(1 for record in records if record.downsampled),
        "headshot_kept": sum(1 for record in records if record.downsampled and record.kept),
        "added_boxes": added_by_class,
    }


def build_label_text(record: ImageRecord) -> str:
    """Pristine original lines followed by synthetic lines, newline-terminated."""
    return "\n".join(record.original_lines + record.synthetic_lines) + "\n"


def write_source_tree(source_name: str, output_root: Path, records: list[ImageRecord], totals: dict) -> None:
    """Write the enriched staging tree for one source (symlink + labels + manifest).

    An empty source still yields its (empty) images/labels dirs and a manifest
    with zero totals, so downstream tooling can rely on the shape. The managed
    ``images/`` and ``labels/`` directories converge to exactly the kept set on
    every run: entries left by a prior run (e.g. a headshot that a different
    ``--seed`` now excludes) are removed, so an ``--override`` rerun can never
    leak stale files into ``unify-stage1`` while the manifest says EXCLUDED.
    Inputs are never touched — only files this writer owns are unlinked.
    """
    images_out = output_root / source_name / "images"
    labels_out = output_root / source_name / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    kept_records = [record for record in records if record.kept and record.image_path is not None]
    expected_images = {record.image_name for record in kept_records}
    expected_labels = {record.label_name for record in kept_records}

    for managed_dir, expected_names in (
        (images_out, expected_images),
        (labels_out, expected_labels),
    ):
        for existing in sorted(managed_dir.iterdir()):
            if existing.name in expected_names:
                continue
            if existing.is_dir():
                logger.warning(
                    "[%s] Unexpected directory in %s left untouched: %s",
                    source_name, managed_dir, existing,
                )
                continue
            existing.unlink()
            logger.debug("[%s] Removed stale output from prior run: %s", source_name, existing)

    for record in kept_records:
        link = images_out / record.image_name
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(str(record.image_path), link)
        (labels_out / record.label_name).write_text(build_label_text(record), encoding="utf-8")

    manifest = {
        "images": {record.image_name: record.manifest_entry() for record in records},
        "totals": totals,
    }
    manifest_path = output_root / source_name / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    logger.info("[%s] Wrote enriched tree -> %s", source_name, output_root / source_name)


# ========================================================================
# CLI  (Law 2 — parse and validate at the boundary)
# ========================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Geometrically verified cross-dataset pseudo-labeling for Stage 1.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=Path, default=Path("config/pipeline.yaml"),
        help="Pipeline config used to resolve the default --staging-dir.",
    )
    parser.add_argument(
        "--target-section",
        choices=["manga109", "anime", "all", "calibrate"], default="all",
        help=(
            "Which staging sources to enrich; 'calibrate' fits the face box "
            "adapter on izutsumi + manga109 instead of enriching outputs."
        ),
    )
    parser.add_argument(
        "--staging-dir", type=Path, default=None,
        help="Staging root holding <source>/images + <source>/labels "
             "(default from config: <data>/data/staging).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Enriched output root (default: sibling of the staging root, "
             "<staging>_enriched). Inputs are never mutated.",
    )
    parser.add_argument(
        "--adapter", type=Path, default=None,
        help="FaceBoxAdapter JSON (REQUIRED for the anime face stage: "
             "shrinks loose face predictions onto the tight Manga109 convention).",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Compute device for teacher inference ('cpu', 'mps', 'cuda').",
    )
    parser.add_argument(
        "--catnip-model", type=Path, default=None,
        help="Catnip Stage 1 checkpoint path (REQUIRED for the manga109 section: "
             "second head teacher fused with deepghs_head_x).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the enrichment statistics without writing anything.",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="Seed for the deterministic headshot-only downsampling draws.",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help=(
            "Cap the number of images processed per source (0 = all; smoke "
            "tests). In calibrate mode it caps the manga109 strided sample "
            "and 0 keeps the 2,000-page plan default (pass >= the page "
            "count for every page)."
        ),
    )
    parser.add_argument(
        "--override", action="store_true",
        help="Allow writing into a non-empty --output-dir (reruns).",
    )
    args = parser.parse_args(argv)

    sources = SECTION_SOURCES[args.target_section]
    needs_manga109 = any(regime == REGIME_MANGA109 for _, regime in sources)
    needs_anime = any(regime == REGIME_ANIME for _, regime in sources)
    is_calibrate = args.target_section == "calibrate"

    if is_calibrate:
        if args.adapter is None:
            parser.error(
                "--adapter is the OUTPUT JSON path for --target-section calibrate "
                "(e.g. config/face_box_adapter.json)"
            )
    else:
        if needs_manga109 and args.catnip_model is None:
            parser.error("--catnip-model is required for --target-section manga109|all")
        if needs_anime and args.adapter is None:
            parser.error("--adapter (FaceBoxAdapter JSON) is required for --target-section anime|all")
    if args.limit < 0:
        parser.error("--limit must be >= 0")

    return args


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """Resolve the staging root and output root from the CLI args/config."""
    if args.staging_dir is not None:
        staging_root = args.staging_dir
    else:
        from src.config import load_settings

        staging_root = load_settings(args.config).paths.data / "data" / "staging"
    output_root = args.output_dir if args.output_dir is not None else staging_root.parent / f"{staging_root.name}_enriched"
    return staging_root, output_root


def _print_summary(per_source: dict[str, dict], staging_root: Path, output_root: Path, dry_run: bool) -> None:
    mode = "DRY RUN — no files written" if dry_run else "COMPLETE"
    print(f"\n{'=' * 60}\n{' Stage 1 Pseudo-Label Enrichment — ' + mode:=^60}\n{'=' * 60}")
    print(f"\n  Staging root:  {staging_root}")
    print(f"  Output root:   {output_root}")
    for source_name in per_source:
        totals = per_source[source_name]
        added = totals["added_boxes"]
        print(f"\n  [{source_name}]")
        print(f"    images seen:      {totals['images_seen']:>6}")
        print(f"    images kept:      {totals['images_kept']:>6}")
        print(f"    images excluded:  {totals['images_excluded']:>6}")
        print(f"    headshot-only:    {totals['headshot_only']:>6} (kept {totals['headshot_kept']})")
        print("    synthetic boxes:")
        for class_key in ("0", "1", "2"):
            label = CLASS_NAMES[int(class_key)]
            print(f"      class {class_key} ({label}): {added[class_key]:>6}")
    print(f"\n{'=' * 60}\n")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-7s %(message)s",
    )

    staging_root, output_root = _resolve_paths(args)
    sources = SECTION_SOURCES[args.target_section]

    missing = [name for name, _ in sources if not (staging_root / name).is_dir()]
    if missing:
        logger.error("Staging sources not found under %s: %s", staging_root, ", ".join(missing))
        sys.exit(1)

    if args.target_section == "calibrate":
        # Adapter fitting is its own flow: no enriched output tree.
        logger.info(
            "Calibrating face box adapter over %s | device=%s | manga109 sample limit=%s",
            [name for name, _ in sources], args.device, args.limit or "all",
        )
        report = calibrate_face_adapter(
            staging_root,
            args.adapter,
            device=args.device,
            limit=args.limit,
            dry_run=args.dry_run,
        )
        _print_calibration_report(report, args.dry_run)
        return

    if not args.dry_run and not args.override and output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(
            f"Output directory {output_root} already exists and is non-empty. "
            f"Pass --override to replace it (inputs are never touched)."
        )

    adapter = FaceBoxAdapter.from_json(args.adapter) if args.adapter is not None else None
    logger.info(
        "Target section '%s' -> %s | device=%s | seed=%d | limit=%s",
        args.target_section,
        [name for name, _ in sources],
        args.device,
        args.seed,
        args.limit or "all",
    )

    per_source: dict[str, dict] = {}
    for source_name, regime in sources:
        logger.info("Enriching [%s] (%s regime) …", source_name, regime)
        per_source[source_name] = process_source(
            source_name=source_name,
            regime=regime,
            staging_root=staging_root,
            output_root=output_root,
            device=args.device,
            catnip_model=args.catnip_model,
            adapter=adapter,
            limit=args.limit,
            seed=args.seed,
            dry_run=args.dry_run,
        )

    _print_summary(per_source, staging_root, output_root, args.dry_run)


if __name__ == "__main__":
    main()
