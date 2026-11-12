"""Dynamic SAHI detection extraction with panel/page gating, WBF, and
hierarchical anatomy filtering (implementation plan tasks 2.2 + 2.3).

Pipeline for one image (all thresholds and sizes come from
``settings.params.sahi``):

1. **Gate** on the input dimensions:
   - ``panel`` (``max(H, W) <= panel_max_dim``): one direct prediction pass on
     the full image (the model letterboxes to 640px internally).
   - ``page``: downscale uniformly so the longer side equals
     ``canonical_page_max_dim`` (never upscale), then predict each
     640x640 window cut by ``sahi.slicing.get_slice_bboxes`` with
     ``auto_slice_resolution=False`` — byte-identical windows to the
     training pre-slicer — plus one full-page pass for global context.
2. **Project** every raw detection back to original-image float pixels
   (slice offset, then ``/ scale``).
3. **Fuse** overlapping predictions per class: Weighted Box Fusion
   (confidence-weighted coordinate averaging) or greedy NMM (largest box
   per cluster).
4. **Clamp** to the image and drop degenerate boxes.
5. **Filter** by per-class calibrated confidence thresholds, then keep
   low-confidence faces only when geometrically contained in a surviving
   head/body box (hierarchical anatomy verification).
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image
from sahi import AutoDetectionModel
from sahi.slicing import get_slice_bboxes
from tqdm import tqdm

from src.config import SahiParams, get_settings

logger = logging.getLogger(__name__)

GATE_PANEL = "panel"
GATE_PAGE = "page"

MATCH_METRIC_IOU = "IOU"
MATCH_METRIC_IOS = "IOS"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

# Hierarchical anatomy filter constants (implementation plan section 2):
# faces at or above hi_conf are retained unconditionally (extreme
# close-ups), faces below lo_conf are always dropped, faces in between
# must be contained (IoS >= ios_threshold) inside a detected head/body.
HIERARCHY_HI_CONF = 0.70
HIERARCHY_LO_CONF = 0.45
HIERARCHY_IOS_THRESHOLD = 0.75
FACE_PARENT_CLASSES = ("head", "body")

# The predictor seam: RGB ndarray -> list of (x1, y1, x2, y2, score, class_id)
# tuples in the pixel space of the array it received.
PredictorFn = Callable[[np.ndarray], list[tuple[float, float, float, float, float, int]]]


@dataclass(frozen=True)
class DetectedPart:
    """One fused detection in original-image pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    class_name: str


@dataclass(frozen=True)
class _RawDetection:
    """A detection in the pixel space of the array it was predicted on."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int


@dataclass
class ExtractionSummary:
    """Result of extracting a directory of images.

    ``crops`` holds one entry per detected part, in per-image detection
    order. ``crop_path`` is the written crop file, or ``None`` when
    ``save_crops=False`` or writing the crop failed.
    """

    images_processed: int = 0
    crops_saved: int = 0
    crops: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Raw detection helpers
# ---------------------------------------------------------------------------


def raw_box(detection: _RawDetection) -> tuple[float, float, float, float]:
    """Raw detection as an (x1, y1, x2, y2) box."""
    return (detection.x1, detection.y1, detection.x2, detection.y2)


def part_box(part: DetectedPart) -> tuple[int, int, int, int]:
    """Detected part as an (x1, y1, x2, y2) box."""
    return (part.x1, part.y1, part.x2, part.y2)


def raw_to_original_space(
    detection: _RawDetection,
    slice_offset_x: float,
    slice_offset_y: float,
    scale: float,
) -> _RawDetection:
    """Project a raw detection into original-image pixel space."""
    x1, y1, x2, y2 = project_box_to_original_space(
        raw_box(detection), slice_offset_x, slice_offset_y, scale
    )
    return _RawDetection(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        score=detection.score,
        class_id=detection.class_id,
    )


# ---------------------------------------------------------------------------
# Pure geometry (all operate on plain coordinates, no I/O)
# ---------------------------------------------------------------------------


def decide_gate(height: int, width: int, panel_max_dim: int) -> str:
    """Return the inference route: ``panel`` (single pass) or ``page`` (sliced)."""
    if panel_max_dim <= 0:
        raise ValueError(f"panel_max_dim must be positive, got {panel_max_dim}")
    if height <= 0 or width <= 0:
        raise ValueError(f"Image dimensions must be positive, got {height}x{width}")
    if max(height, width) <= panel_max_dim:
        return GATE_PANEL
    return GATE_PAGE


def compute_page_scale(height: int, width: int, canonical_page_max_dim: int) -> float:
    """Downscale factor for a page; 1.0 whenever the page fits the canonical size."""
    if canonical_page_max_dim <= 0:
        raise ValueError(
            f"canonical_page_max_dim must be positive, got {canonical_page_max_dim}"
        )
    longest_side = max(height, width)
    if longest_side <= canonical_page_max_dim:
        return 1.0
    return canonical_page_max_dim / longest_side


def compute_page_dimensions(height: int, width: int, scale: float) -> tuple[int, int]:
    """Downscaled (height, width) for a page gate; never rounds below 1px."""
    if not 0.0 < scale <= 1.0:
        raise ValueError(f"Page scale must be in ]0, 1.0], got {scale}")
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def project_box_to_original_space(
    box: tuple[float, float, float, float],
    slice_offset_x: float,
    slice_offset_y: float,
    scale: float,
) -> tuple[float, float, float, float]:
    """Map a slice-local box to original-image pixels.

    ``scale`` is the page downscale factor applied before slicing, so the
    reverse projection divides by it (``scale == 1`` leaves the box intact).
    """
    if not 0.0 < scale <= 1.0:
        raise ValueError(f"Projection scale must be in ]0, 1.0], got {scale}")
    x1, y1, x2, y2 = box
    return (
        (slice_offset_x + x1) / scale,
        (slice_offset_y + y1) / scale,
        (slice_offset_x + x2) / scale,
        (slice_offset_y + y2) / scale,
    )


def clamp_box_to_image(
    box: tuple[float, float, float, float], height: int, width: int
) -> tuple[int, int, int, int] | None:
    """Clamp a float box to the image; return ``None`` for degenerate boxes."""
    x1, y1, x2, y2 = box
    if not all(np.isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    clamped = (
        min(max(0.0, x1), float(width)),
        min(max(0.0, y1), float(height)),
        min(max(0.0, x2), float(width)),
        min(max(0.0, y2), float(height)),
    )
    ix1, iy1, ix2, iy2 = (int(v) for v in clamped)
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return (ix1, iy1, ix2, iy2)


def box_area(box: tuple[float, float, float, float]) -> float:
    """Area of an (x1, y1, x2, y2) box; 0.0 for degenerate boxes."""
    dx, dy = box[2] - box[0], box[3] - box[1]
    if dx <= 0 or dy <= 0:
        return 0.0
    return dx * dy


def intersection_area(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    """Area of the intersection of two boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def intersection_over_union(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    """IoU of two boxes (0.0 when the union is degenerate)."""
    union = box_area(a) + box_area(b) - intersection_area(a, b)
    if union <= 0.0:
        return 0.0
    return intersection_area(a, b) / union


def intersection_over_self(
    box: tuple[float, float, float, float],
    reference: tuple[float, float, float, float],
) -> float:
    """Intersection area divided by ``box``'s own area (containment ratio)."""
    self_area = box_area(box)
    if self_area <= 0.0:
        return 0.0
    return intersection_area(box, reference) / self_area


def validate_match_metric(metric: str) -> str:
    """Normalize a match metric to its canonical form; raise for unknown names."""
    normalized = metric.strip().upper()
    if normalized not in (MATCH_METRIC_IOU, MATCH_METRIC_IOS):
        raise ValueError(f"Unknown match metric {metric!r} (expected 'IOU' or 'IOS')")
    return normalized


def compute_match_metric(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    metric: str,
) -> float:
    """Similarity of two boxes under ``"IOU"`` or ``"IOS"`` (case-insensitive)."""
    normalized = validate_match_metric(metric)
    if normalized == MATCH_METRIC_IOU:
        return intersection_over_union(a, b)
    # IOS is symmetric: intersection over the smaller of the two areas.
    return max(intersection_over_self(a, b), intersection_over_self(b, a))


# ---------------------------------------------------------------------------
# Fusion (in-repo; sahi 0.11.x ships no WBF postprocessor)
# ---------------------------------------------------------------------------


def weighted_box_fusion(
    boxes: list[tuple[float, float, float, float]],
    scores: list[float],
    match_metric: str = MATCH_METRIC_IOS,
    match_threshold: float = 0.40,
) -> tuple[list[tuple[float, float, float, float]], list[float]]:
    """Fuse overlapping boxes via Solovyev-style Weighted Box Fusion.

    Candidates are consumed in descending score order (stable on ties). A
    candidate joins the first cluster whose fused box is similar enough
    (``match_metric`` >= ``match_threshold``); otherwise it opens a new
    cluster. A cluster's fused box is the confidence-weighted mean of its
    member coordinates; its fused score is the mean of the member scores
    clipped to [0, 1]. Both lists are returned in cluster formation order.
    """
    if len(boxes) != len(scores):
        raise ValueError(
            f"boxes and scores must have equal length, got {len(boxes)} != {len(scores)}"
        )
    if not boxes:
        return [], []
    validate_match_metric(match_metric)

    clusters: list[dict[str, list]] = []
    for box, score in _by_score_descending(boxes, scores):
        target = next(
            (
                cluster
                for cluster in clusters
                if compute_match_metric(box, _cluster_box(cluster), match_metric)
                >= match_threshold
            ),
            None,
        )
        if target is None:
            clusters.append({"boxes": [box], "scores": [score]})
        else:
            target["boxes"].append(box)
            target["scores"].append(score)

    fused_boxes = [_cluster_box(cluster) for cluster in clusters]
    fused_scores = [
        min(1.0, max(0.0, sum(cluster["scores"]) / len(cluster["scores"])))
        for cluster in clusters
    ]
    return fused_boxes, fused_scores


def _by_score_descending(
    boxes: list[tuple[float, float, float, float]], scores: list[float]
) -> list[tuple[tuple[float, float, float, float], float]]:
    """(box, score) pairs ordered by score descending, stable on ties."""
    indexed = list(enumerate(zip(boxes, scores)))
    indexed.sort(key=lambda item: (-item[1][1], item[0]))
    return [pair for _, pair in indexed]


def _cluster_box(cluster: dict[str, list]) -> tuple[float, float, float, float]:
    """Confidence-weighted mean of a cluster's member coordinates."""
    total_weight = sum(cluster["scores"])
    if total_weight <= 0.0:
        count = len(cluster["boxes"])
        return tuple(
            sum(box[i] for box in cluster["boxes"]) / count for i in range(4)
        )  # type: ignore[return-value]
    return tuple(  # type: ignore[return-value]
        sum(box[i] * score for box, score in zip(cluster["boxes"], cluster["scores"]))
        / total_weight
        for i in range(4)
    )


def nmm_fusion(
    boxes: list[tuple[float, float, float, float]],
    scores: list[float],
    match_metric: str = MATCH_METRIC_IOS,
    match_threshold: float = 0.40,
) -> tuple[list[tuple[float, float, float, float]], list[float]]:
    """Greedy Non-Maximum Merging: keep the largest box of each cluster.

    Boxes are consumed in descending area order (then descending score,
    stable on ties). A box is suppressed when it matches any already-kept
    box under ``match_metric`` >= ``match_threshold``; kept boxes keep
    their original coordinates and scores.
    """
    if len(boxes) != len(scores):
        raise ValueError(
            f"boxes and scores must have equal length, got {len(boxes)} != {len(scores)}"
        )
    if not boxes:
        return [], []
    validate_match_metric(match_metric)

    indexed = list(enumerate(zip(boxes, scores)))
    indexed.sort(key=lambda item: (-box_area(item[1][0]), -item[1][1], item[0]))

    kept: list[tuple[tuple[float, float, float, float], float]] = []
    for _, (box, score) in indexed:
        if all(
            compute_match_metric(box, kept_box, match_metric) < match_threshold
            for kept_box, _ in kept
        ):
            kept.append((box, score))
    return [box for box, _ in kept], [score for _, score in kept]


def fuse_detections_per_class(
    detections: list[_RawDetection],
    postprocess_type: str,
    match_metric: str,
    match_threshold: float,
) -> list[_RawDetection]:
    """Fuse raw detections independently per class; classes never merge."""
    fusion: Callable = {"WBF": weighted_box_fusion, "NMM": nmm_fusion}.get(
        postprocess_type
    )
    if fusion is None:
        raise ValueError(
            f"Unknown postprocess_type {postprocess_type!r} (expected 'WBF' or 'NMM')"
        )

    detections_by_class: dict[int, list[_RawDetection]] = {}
    for detection in detections:
        detections_by_class.setdefault(detection.class_id, []).append(detection)

    fused: list[_RawDetection] = []
    for class_id, group in detections_by_class.items():
        fused_boxes, fused_scores = fusion(
            [raw_box(detection) for detection in group],
            [detection.score for detection in group],
            match_metric=match_metric,
            match_threshold=match_threshold,
        )
        fused.extend(
            _RawDetection(
                x1=x1, y1=y1, x2=x2, y2=y2, score=score, class_id=class_id
            )
            for (x1, y1, x2, y2), score in zip(fused_boxes, fused_scores)
        )
    return fused


# ---------------------------------------------------------------------------
# Filtering (pure, applied to DetectedPart lists)
# ---------------------------------------------------------------------------


def apply_class_confidence_thresholds(
    detections: list[DetectedPart],
    class_conf_thresholds: dict[str, float],
    exempt_faces: bool = False,
) -> list[DetectedPart]:
    """Keep detections whose score meets their class's calibrated threshold.

    ``exempt_faces=True`` defers face filtering to the hierarchy rule: the
    threshold lookup still validates every class name (fail loud on unknown
    classes), but face scores are not gated. The hierarchy band
    ``[lo_conf, hi_conf)`` + containment IS the calibrated face filter
    (spec §2), so gating faces to a class threshold first would render the
    ``[threshold, lo_conf)`` stretch unreachable when the hierarchy runs.
    Callers enabling ``hierarchical_verify`` must pass True.
    """
    kept: list[DetectedPart] = []
    for detection in detections:
        threshold = class_conf_thresholds.get(detection.class_name)
        if threshold is None:
            raise ValueError(
                f"No confidence threshold configured for class {detection.class_name!r}; "
                f"known classes: {sorted(class_conf_thresholds)}"
            )
        if exempt_faces and detection.class_name == "face":
            kept.append(detection)
            continue
        if detection.score >= threshold:
            kept.append(detection)
    return kept


def filter_anatomical_hierarchy(
    detections: list[DetectedPart],
    enabled: bool = True,
    hi_conf: float = HIERARCHY_HI_CONF,
    lo_conf: float = HIERARCHY_LO_CONF,
    ios_threshold: float = HIERARCHY_IOS_THRESHOLD,
) -> list[DetectedPart]:
    """Verify low-confidence faces against surviving head/body detections.

    - ``face`` with ``score >= hi_conf``: kept unconditionally (extreme
      close-up panels where only a face fills the frame).
    - ``face`` with ``lo_conf <= score < hi_conf``: kept only when
      contained inside a detected ``head``/``body`` box with
      ``IoS(face, parent) >= ios_threshold``.
    - ``face`` with ``score < lo_conf``: dropped.
    - Non-face classes pass through untouched.
    """
    if not enabled:
        return list(detections)

    parents = [
        part for part in detections if part.class_name in FACE_PARENT_CLASSES
    ]
    kept: list[DetectedPart] = []
    for detection in detections:
        if detection.class_name != "face":
            kept.append(detection)
            continue
        if detection.score >= hi_conf:
            kept.append(detection)
            continue
        if detection.score < lo_conf:
            continue
        max_containment = max(
            (
                intersection_over_self(part_box(detection), part_box(parent))
                for parent in parents
            ),
            default=0.0,
        )
        if max_containment >= ios_threshold:
            kept.append(detection)
    return kept


# ---------------------------------------------------------------------------
# SahiExtractor
# ---------------------------------------------------------------------------


class SahiExtractor:
    """Detects body/head/face parts via gated SAHI inference (plan tasks 2.2/2.3)."""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        sahi_params: SahiParams | None = None,
        predictor: PredictorFn | None = None,
    ) -> None:
        """Load the Stage 1 YOLO model for gated SAHI inference.

        Args:
            model_path: Path to the Stage 1 YOLO weights (unused when
                ``predictor`` is given).
            device: ``"cpu"``, ``"mps"``, or ``"cuda"``.
            sahi_params: Override the pipeline defaults; ``None`` uses
                ``get_settings().params.sahi``.
            predictor: Test seam. When given, ``AutoDetectionModel`` is not
                loaded and this callable replaces every inference pass. It
                receives an RGB ``np.ndarray`` and returns
                ``(x1, y1, x2, y2, score, class_id)`` sequences in the
                pixel space of that array.
        """
        if sahi_params is not None and not isinstance(sahi_params, SahiParams):
            raise TypeError(
                f"sahi_params must be a SahiParams instance, got {type(sahi_params)!r}"
            )
        if predictor is not None and not callable(predictor):
            raise TypeError(f"predictor must be callable, got {type(predictor)!r}")

        self.model_path = model_path
        self.device = device
        self.params = sahi_params if sahi_params is not None else get_settings().params.sahi
        self.class_id_to_name: dict[int, str] = {
            class_id: name
            for name, class_id in vars(get_settings().labels.stage1).items()
        }

        self._predictor = predictor
        self._detection_model = None
        if predictor is None:
            logger.info("Loading YOLO model from %s onto %s...", model_path, device)
            self._detection_model = AutoDetectionModel.from_pretrained(
                model_type="yolov8",
                model_path=model_path,
                confidence_threshold=self.params.confidence_threshold,
                device=device,
            )

    def extract_from_image(
        self, image_input: str | Path | np.ndarray
    ) -> list[DetectedPart]:
        """Detect parts on one image; fused detections in original pixels.

        Accepts either a file path (str or Path) or an image array (np.ndarray).
        """
        if isinstance(image_input, (str, Path)):
            image_rgb = self._read_image_rgb(image_input)
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                image_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.ndim == 3 and image_input.shape[2] == 3:
                # Standard OpenCV cv2.imread produces BGR; convert to RGB
                image_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            elif image_input.ndim == 3 and image_input.shape[2] == 4:
                image_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGRA2RGB)
            else:
                image_rgb = image_input
        else:
            raise TypeError(
                f"Expected str, Path, or np.ndarray, got {type(image_input)!r}"
            )

        height, width = image_rgb.shape[:2]
        if height == 0 or width == 0:
            return []

        route = decide_gate(height, width, self.params.panel_max_dim)
        if route == GATE_PANEL:
            # Direct single pass: predictions already live in original pixels.
            raw_detections = self._detect(image_rgb)
        else:
            raw_detections = self._predict_page(image_rgb)

        fused = fuse_detections_per_class(
            raw_detections,
            postprocess_type=self.params.postprocess_type,
            match_metric=self.params.postprocess_match_metric,
            match_threshold=self.params.postprocess_match_threshold,
        )
        parts = self._to_detected_parts(fused, height, width)
        parts = apply_class_confidence_thresholds(
            parts,
            self.params.class_conf_thresholds,
            exempt_faces=self.params.hierarchical_verify,
        )
        return filter_anatomical_hierarchy(parts, self.params.hierarchical_verify)

    def extract_from_directory(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        save_crops: bool = True,
    ) -> ExtractionSummary:
        """Detect parts on every image below ``input_dir`` and save crops.

        Images are scanned recursively for jpg/jpeg/png/webp. An image
        that fails to decode or predict is logged and skipped so one bad
        file cannot abort the batch. Crops are written as
        ``{stem}_crop_{i}.jpg`` (``i`` = per-image detection index); empty
        crops are skipped.
        """
        image_paths = find_image_files(Path(input_dir))
        output_dir = Path(output_dir)
        if save_crops:
            output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Found %d images to process.", len(image_paths))
        summary = ExtractionSummary()

        for image_path in tqdm(image_paths, desc="Extracting via SAHI"):
            try:
                parts = self.extract_from_image(image_path)
            except Exception as exc:  # noqa: BLE001 - batch-level resilience
                logger.warning("Skipping %s (%s)", image_path, exc)
                continue

            image_bgr = cv2.imread(str(image_path)) if save_crops else None
            if save_crops and image_bgr is None:
                logger.warning(
                    "Could not re-read %s for cropping; crop writes skipped.", image_path
                )

            summary.images_processed += 1
            for crop_index, part in enumerate(parts):
                crop_path = None
                if save_crops and image_bgr is not None:
                    crop_path = self._save_crop(
                        image_bgr, part, output_dir, image_path.stem, crop_index
                    )
                    if crop_path is not None:
                        summary.crops_saved += 1
                summary.crops.append(
                    {
                        "image": str(image_path),
                        "crop_path": str(crop_path) if crop_path is not None else None,
                        "bbox": [part.x1, part.y1, part.x2, part.y2],
                        "class_name": part.class_name,
                        "score": part.score,
                    }
                )

        logger.info(
            "Extraction complete! Detected %d parts (%d crops saved) under %s",
            len(summary.crops),
            summary.crops_saved,
            output_dir,
        )
        return summary

    # ------------------------------------------------------------------
    # Image I/O and inference plumbing
    # ------------------------------------------------------------------

    def _read_image_rgb(self, image_path: str | Path) -> np.ndarray:
        """Decode an image into a 3-channel RGB array (fail loud on error)."""
        with Image.open(image_path) as pil_image:
            pil_image.load()
            return np.asarray(pil_image.convert("RGB"))

    def _predict_page(self, image_rgb: np.ndarray) -> list[_RawDetection]:
        """Predict a full page: canonical rescale, per-slice + full-page passes.

        The extra full-page pass (beyond the plan's slicing spec) preserves
        the predecessor extractor's ``perform_standard_pred=True`` global
        context: page-wide prediction catches characters that fragment
        across slice seams. The same object seen by a slice and by the
        full-page pass yields a high-IOS pair that WBF merges; genuinely
        disjoint duplicates would both persist, but such cross-scale
        ghost detections have not been observed at these thresholds. The
        cost is one extra 640px inference per page.
        """
        height, width = image_rgb.shape[:2]
        scale = compute_page_scale(height, width, self.params.canonical_page_max_dim)
        page_rgb = rescale_image(image_rgb, scale)
        page_height, page_width = page_rgb.shape[:2]

        slice_windows = get_slice_bboxes(
            image_height=page_height,
            image_width=page_width,
            slice_height=self.params.slice_height,
            slice_width=self.params.slice_width,
            overlap_height_ratio=self.params.overlap_ratio,
            overlap_width_ratio=self.params.overlap_ratio,
            auto_slice_resolution=False,
        )

        raw_detections: list[_RawDetection] = []
        for slice_x1, slice_y1, slice_x2, slice_y2 in slice_windows:
            slice_rgb = np.ascontiguousarray(
                page_rgb[slice_y1:slice_y2, slice_x1:slice_x2]
            )
            for detection in self._detect(slice_rgb):
                raw_detections.append(
                    raw_to_original_space(detection, slice_x1, slice_y1, scale)
                )

        # Full-page pass: global context catches characters that are small in
        # every 640px window but large relative to the downscaled page.
        for detection in self._detect(page_rgb):
            raw_detections.append(raw_to_original_space(detection, 0.0, 0.0, scale))
        return raw_detections

    def _detect(self, image_rgb: np.ndarray) -> list[_RawDetection]:
        """One forward pass on an RGB array; detections in that array's pixels."""
        if self._predictor is not None:
            return _parse_predictor_output(self._predictor(image_rgb))

        if self._detection_model is None:
            raise RuntimeError("Neither a predictor nor a detection model is loaded.")
        self._detection_model.perform_inference(image_rgb)
        self._detection_model.convert_original_predictions()
        return [
            _RawDetection(
                x1=prediction.bbox.minx,
                y1=prediction.bbox.miny,
                x2=prediction.bbox.maxx,
                y2=prediction.bbox.maxy,
                score=prediction.score.value,
                class_id=int(prediction.category.id),
            )
            for prediction in self._detection_model.object_prediction_list
        ]

    def _to_detected_parts(
        self, detections: list[_RawDetection], height: int, width: int
    ) -> list[DetectedPart]:
        """Clamp fused detections to the image and resolve class names."""
        parts: list[DetectedPart] = []
        for detection in detections:
            clamped = clamp_box_to_image(raw_box(detection), height, width)
            if clamped is None:
                continue
            class_name = self.class_id_to_name.get(detection.class_id)
            if class_name is None:
                raise ValueError(
                    f"Model predicted unknown class id {detection.class_id}; "
                    f"known classes: {self.class_id_to_name}"
                )
            x1, y1, x2, y2 = clamped
            parts.append(
                DetectedPart(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    score=detection.score, class_name=class_name,
                )
            )
        return parts

    def _save_crop(
        self,
        image_bgr: np.ndarray,
        part: DetectedPart,
        output_dir: Path,
        image_stem: str,
        crop_index: int,
    ) -> Path | None:
        """Write ``{stem}_crop_{i}.jpg`` and return its path (``None`` when empty)."""
        crop = image_bgr[part.y1:part.y2, part.x1:part.x2]
        if crop.size == 0:
            return None
        save_path = output_dir / f"{image_stem}_crop_{crop_index}.jpg"
        if not cv2.imwrite(str(save_path), crop):
            logger.warning("cv2.imwrite failed for %s", save_path)
            return None
        return save_path


# ---------------------------------------------------------------------------
# Module-level image utilities
# ---------------------------------------------------------------------------


def rescale_image(image_rgb: np.ndarray, scale: float) -> np.ndarray:
    """Downscale an RGB image by ``scale`` (unchanged when ``scale == 1.0``)."""
    if not 0.0 < scale <= 1.0:
        raise ValueError(f"Scale must be in ]0, 1.0], got {scale}")
    if scale == 1.0:
        return image_rgb
    height, width = image_rgb.shape[:2]
    new_height, new_width = compute_page_dimensions(height, width, scale)
    return cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)


def find_image_files(input_dir: Path) -> list[Path]:
    """Recursive, deduplicated, sorted image listing below ``input_dir``.

    Extension matching is case-insensitive so ``.JPG`` rips are picked up too.
    """
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _parse_predictor_output(output: object) -> list[_RawDetection]:
    """Validate predictor output into trusted ``_RawDetection`` state."""
    raw_detections: list[_RawDetection] = []
    for index, item in enumerate(output):  # type: ignore[arg-type]
        if not hasattr(item, "__len__") or len(item) != 6:
            raise ValueError(
                f"Predictor item {index} must be (x1, y1, x2, y2, score, class_id), "
                f"got: {item!r}"
            )
        x1, y1, x2, y2, score, class_id = item
        try:
            score = float(score)
            class_id = int(class_id)
            parsed = _RawDetection(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                score=score,
                class_id=class_id,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Predictor item {index} has non-numeric values, got: {item!r}"
            ) from exc
        if not (0.0 <= score <= 1.0):
            raise ValueError(
                f"Predictor item {index} has confidence {score} outside [0, 1]: "
                f"{item!r}"
            )
        raw_detections.append(parsed)
    return raw_detections
