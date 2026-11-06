#!/usr/bin/env python3
"""
Catnip CLI — Character Re-Identification in manga.
"""

import argparse
import json
import logging
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from src.config import Settings

logger = logging.getLogger("catnip")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _apply_overrides(config_path: Path, overrides: list[str]) -> Settings:
    """Merge dot-notation KEY=VALUE overrides into OmegaConf, then re-validate with Pydantic."""
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    cfg = OmegaConf.load(config_path)
    valid_keys = set(Settings.model_fields.keys())

    for override in overrides:
        if "=" not in override:
            logger.error(f"Invalid override format: '{override}'. Expected KEY=VALUE (dot-notation).")
            sys.exit(1)
        key, value = override.split("=", 1)

        root_key = key.split(".")[0]
        if root_key not in valid_keys:
            logger.error(f"Invalid override key '{key}': no top-level key '{root_key}'")
            logger.error(f"Available top-level keys: {', '.join(sorted(valid_keys))}")
            sys.exit(1)

        try:
            if value.lower() in ("true", "false"):
                typed_value = value.lower() == "true"
            else:
                try:
                    typed_value = int(value)
                except ValueError:
                    try:
                        typed_value = float(value)
                    except ValueError:
                        typed_value = value
        except Exception:
            typed_value = value

        try:
            OmegaConf.update(cfg, key, typed_value)
        except Exception as e:
            logger.error(f"Invalid override key '{key}': {e}")
            logger.error(f"Available top-level keys: {', '.join(sorted(valid_keys))}")
            sys.exit(1)

    try:
        resolved = OmegaConf.to_container(cfg, resolve=True)
        return Settings(**resolved)
    except Exception as e:
        logger.error(f"Invalid value for override: {e}")
        sys.exit(1)


def _setup_logging(verbose: bool = False):
    """Configure logging with tqdm-compatible formatting."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)-8s %(name)-20s %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )


# ---------------------------------------------------------------------------
# Model checkpoint resolution (mirrors src/eval/adapters.get_detector("catnip"))
# ---------------------------------------------------------------------------


def _stage1_checkpoint_candidates(settings: Settings) -> list[Path]:
    """Candidate list mirroring the Catnip branch of adapters.get_detector.

    Kept inline (instead of importing the benchmark stack) so the data
    pipeline CLI stays independent of evaluation-only dependencies.
    """
    return [
        # 1. Full training deliverable (Colab / production run)
        settings.paths.model_dir / "yolo26_stage1_body_head_face.pt",
        # 2. Raw best weights from the full training run
        settings.paths.runs_dir / "detect" / "stage1" / "weights" / "best.pt",
        # 3. Last rolling checkpoint from the full training run
        settings.paths.runs_dir / "detect" / "stage1" / "weights" / "last.pt",
        # 4. Local repository models directory (full model)
        _REPO_ROOT / "models" / "yolo26_stage1_body_head_face.pt",
        # 5. Local repository models directory (M3 subset baseline deliverable)
        _REPO_ROOT / "models" / "yolo26_stage1_body_head_face_m3.pt",
        # 6. M3 subset run weights
        settings.paths.runs_dir / "detect" / "stage1_m3" / "weights" / "best.pt",
        _REPO_ROOT / "runs" / "detect" / "stage1_m3" / "weights" / "best.pt",
        # 7. Cached benchmark model
        Path.home() / ".cache" / "catnip" / "benchmarks" / "catnip_stage1.pt",
    ]


def _stage2_checkpoint_candidates(settings: Settings) -> list[Path]:
    """Candidate list for the Stage 2 character embedder checkpoint."""
    model_name = settings.training.stage2.model
    return [
        settings.paths.model_dir / model_name,
        _REPO_ROOT / "models" / model_name,
        Path.home() / ".cache" / "catnip" / "benchmarks" / model_name,
    ]


def _resolve_checkpoint(settings: Settings, explicit_model, candidates: list[Path], stage: str) -> Path:
    """Resolve a checkpoint: explicit --model when given, else first candidate."""
    if explicit_model is not None:
        model_path = Path(explicit_model)
        if not model_path.is_file():
            logger.error(f"{stage} checkpoint not found: {model_path}")
            sys.exit(1)
        return model_path

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    search_list = "\n  - ".join(str(candidate) for candidate in candidates)
    logger.error(
        f"No {stage} checkpoint found. Pass --model or place the checkpoint at one of:\n  - {search_list}"
    )
    sys.exit(1)


def _resolve_stage1_checkpoint(settings: Settings, explicit_model) -> Path:
    return _resolve_checkpoint(
        settings, explicit_model, _stage1_checkpoint_candidates(settings), "Stage 1"
    )


def _resolve_stage2_checkpoint(settings: Settings, explicit_model) -> Path:
    return _resolve_checkpoint(
        settings, explicit_model, _stage2_checkpoint_candidates(settings), "Stage 2"
    )


# ---------------------------------------------------------------------------
# Stage 1: extract (SAHI)
# ---------------------------------------------------------------------------


def _run_extract(settings: Settings, input_dir: Path, output_dir: Path, model_path: Path) -> Path:
    """Run SAHI extraction and write extract_manifest.json. Returns its path."""
    logger.info("Extract: SAHI sliced inference (Stage 1)")
    logger.info(f"  Input:  {input_dir}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Model:  {model_path}")
    logger.info(f"  Device: {settings.params.device}")

    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    # Imported INSIDE the function so tests can monkeypatch the module
    # attribute (src.detection.sahi_extractor.SahiExtractor) at call time.
    from src.detection.sahi_extractor import SahiExtractor

    crops_dir = Path(output_dir) / "crops"
    try:
        extractor = SahiExtractor(model_path=str(model_path), device=settings.params.device)
        summary = extractor.extract_from_directory(
            input_dir=str(input_dir), output_dir=str(crops_dir), save_crops=True
        )
        manifest = {
            "images_processed": int(summary.images_processed),
            "crops_saved": int(summary.crops_saved),
            "crops": list(summary.crops),
        }
    except Exception as err:
        logger.error(f"Stage 1 extraction failed: {err}")
        sys.exit(1)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "extract_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(
        f"Extract complete: {manifest['crops_saved']} crop(s) from "
        f"{manifest['images_processed']} image(s) -> {manifest_path}"
    )
    return manifest_path


def cmd_extract(args):
    """Stage 1: SAHI body/head/face detection on an image directory."""
    settings = _apply_overrides(args.config, args.override)
    _setup_logging(args.verbose)

    output_dir = Path(args.output_dir) if args.output_dir else Path(settings.paths.crops_dir)
    model_path = _resolve_stage1_checkpoint(settings, args.model)
    _run_extract(settings, Path(args.input_dir), output_dir, model_path)


# ---------------------------------------------------------------------------
# Stage 2: embed (character instances -> fused embeddings)
# ---------------------------------------------------------------------------


def _seed_images(seed_dir: Path, label: str, embedder, add_seed) -> int:
    """Embed every seed crop in ``seed_dir`` as a single-part instance."""
    seed_dir = Path(seed_dir)
    if not seed_dir.is_dir():
        logger.error(f"{label} seed directory not found: {seed_dir}")
        sys.exit(1)
    images = sorted(p for p in seed_dir.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)
    if not images:
        logger.error(f"No {label} seed images (.jpg/.jpeg/.png) found in {seed_dir}")
        sys.exit(1)
    for image in images:
        add_seed(_embed_single_normalized(embedder, image))
    return len(images)


def _embed_single_normalized(embedder, image_path: Path) -> np.ndarray:
    """One crop image file -> L2-normalized float64 embedding (single-part path)."""
    import cv2  # lazy: only the embed/match commands pay the OpenCV import

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        logger.error(f"Unreadable crop image: {image_path}")
        sys.exit(1)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    embedding = embedder.embed_single(image_rgb)
    return _l2_normalize(embedding)


def _l2_normalize(embedding) -> np.ndarray:
    from src.reid.clustering import l2_normalize

    return l2_normalize(np.asarray(embedding, dtype=np.float64).reshape(-1))


def _member_value(obj, names: tuple[str, ...]):
    """Read the first present member. Builder D ships dataclasses, tests ship dicts."""
    for name in names:
        if isinstance(obj, dict):
            if name in obj:
                return obj[name]
        elif hasattr(obj, name):
            return getattr(obj, name)
    return None


def _crop_paths_by_part(manifest_crops: list) -> dict:
    """Index saved crop paths by (image, class_name, bbox) manifest identity.

    CharacterInstance.parts (src/reid/dataset.py) carries only {bbox, score}
    per part, so linked instances recover their crop files by matching the
    original manifest detection — stable because both sides come from the
    same extraction manifest.
    """
    index: dict = {}
    for entry in manifest_crops:
        key = (str(entry["image"]), str(entry["class_name"]), tuple(entry["bbox"]))
        index.setdefault(key, entry.get("crop_path"))
    return index


def _read_instance(instance, position: int, crop_paths_by_part: dict) -> dict:
    """Normalize one link_part_detections instance into trusted typed state.

    CharacterInstance contract (src/reid/dataset.py): instance_id
    "<image>#<idx>", image_path, parts = {class_name: {"bbox", "score"}}.
    Page/panel metadata is absent there, so it defaults to the source image —
    exactly the documented no-panel-segmentation behavior.
    """
    instance_id = _member_value(instance, ("instance_id", "id"))
    image = _member_value(instance, ("image_path", "image", "source_image"))
    parts = _member_value(instance, ("parts",))
    if instance_id is None:
        logger.error(
            f"Stage 2 instance at position {position} exposes neither "
            f"'instance_id' nor 'id' — cannot identify character instances"
        )
        sys.exit(1)
    if image is None:
        logger.error(
            f"Stage 2 instance '{instance_id}' exposes neither 'image_path' "
            "nor 'image' — cannot resolve its part crops"
        )
        sys.exit(1)
    if not isinstance(parts, Mapping) or not parts:
        logger.error(
            f"Stage 2 instance '{instance_id}' (position {position}) exposes "
            "no 'parts' mapping of class_name -> {bbox, score}"
        )
        sys.exit(1)

    normalized_parts = []
    for class_name, part in parts.items():
        bbox = _member_value(part, ("bbox",))
        score = _member_value(part, ("score",))
        if bbox is None:
            logger.error(
                f"Part '{class_name}' of instance '{instance_id}' carries no bbox"
            )
            sys.exit(1)
        crop_path = crop_paths_by_part.get((str(image), str(class_name), tuple(bbox)))
        if not crop_path:
            logger.error(
                f"Embed aborted: no saved crop in extract_manifest.json matches "
                f"part '{class_name}' of instance '{instance_id}' (bbox {bbox}). "
                "Re-run 'catnip extract' with save_crops=True."
            )
            sys.exit(1)
        # A matching entry whose crop FILE is missing stays an INVALID STATE:
        # never half-embed an instance (checked by _embed_instance_parts).
        normalized_parts.append(
            {
                "class_name": str(class_name),
                "crop_path": str(crop_path),
                "score": float(score) if score is not None else 1.0,
                "bbox": [float(c) for c in bbox],
            }
        )

    image = str(image)
    page = _member_value(instance, ("page",)) or image
    panel = _member_value(instance, ("panel",)) or page
    return {
        "instance_id": str(instance_id),
        "image": image,
        "page": str(page),
        "panel": str(panel),
        "parts": normalized_parts,
        "bbox": _instance_bbox(normalized_parts),
    }


def _instance_bbox(parts: list[dict]) -> list[float] | None:
    """Outermost part box of a linked instance (body > head > face).

    The outer box is what Same-Panel Cannot-Link compares: face ⊂ head ⊂
    body, so the body box IS the character's spatial footprint. Instances
    without any of the known classes return None and fall back to the
    documented treat-as-distinct behavior in the panel rule.
    """
    for class_name in ("body", "head", "face"):
        for part in parts:
            if part["class_name"] == class_name:
                return part["bbox"]
    return None


def _load_manifest(input_dir: Path) -> dict:
    """Load extract_manifest.json from input_dir, or {} when absent."""
    manifest_path = Path(input_dir) / "extract_manifest.json"
    if not manifest_path.is_file():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        logger.error(f"Invalid manifest JSON at {manifest_path}: {err}")
        sys.exit(1)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("crops"), list):
        logger.error(
            f"Manifest at {manifest_path} must be an object with a 'crops' list entry"
        )
        sys.exit(1)
    return manifest


def _embed_instance_parts(embedder, fuse_part_embeddings, part_weights: dict, instance: dict) -> np.ndarray:
    """Embed every linked part crop and fuse into one instance embedding."""
    part_embeddings: dict[str, np.ndarray] = {}
    part_scores: dict[str, float] = {}
    for part in instance["parts"]:
        crop_path = Path(part["crop_path"])
        if not crop_path.is_file():
            # A missing crop file is an INVALID STATE: the manifest references
            # crops that must exist on disk. Never half-embed an instance.
            logger.error(
                f"Embed aborted: crop file for part '{part['class_name']}' of "
                f"instance '{instance['instance_id']}' is missing: {crop_path}. "
                "Re-run 'catnip extract' or repair the manifest."
            )
            sys.exit(1)
        part_embeddings[part["class_name"]] = _embed_single_normalized(embedder, crop_path)
        part_scores[part["class_name"]] = part["score"]
    fused = fuse_part_embeddings(
        parts=part_embeddings,
        scores=part_scores,
        weights=part_weights,
    )
    return _l2_normalize(fused)


def _run_embed(settings: Settings, input_dir: Path, output_dir: Path, model_path: Path) -> tuple[Path, Path]:
    """Compute instance embeddings; write embeddings.npy + records.json."""
    logger.info("Embed: computing ReID embeddings (Stage 2)")
    logger.info(f"  Input:  {input_dir}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Model:  {model_path}")

    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    try:
        from src.reid.dataset import link_part_detections
        from src.reid.model import fuse_part_embeddings, load_embedder
    except Exception as err:
        logger.error(
            f"Stage 2 reid modules unavailable: {err}. "
            "src/reid/model.py and src/reid/dataset.py must be present to embed."
        )
        sys.exit(1)

    try:
        embedder = load_embedder(str(model_path))
        manifest = _load_manifest(input_dir)
        part_weights = dict(settings.params.reid.part_weights)

        records: list[dict] = []
        embeddings: list[np.ndarray] = []
        if manifest:
            instances = link_part_detections(manifest["crops"])
            crop_paths_by_part = _crop_paths_by_part(manifest["crops"])
            for position, instance in enumerate(instances):
                normalized = _read_instance(instance, position, crop_paths_by_part)
                embeddings.append(
                    _embed_instance_parts(embedder, fuse_part_embeddings, part_weights, normalized)
                )
                record = {
                    "instance_id": normalized["instance_id"],
                    "image": normalized["image"],
                    "page": normalized["page"],
                    "panel": normalized["panel"],
                    "crop_paths": [part["crop_path"] for part in normalized["parts"]],
                    "parts": [
                        {
                            "class_name": part["class_name"],
                            "crop_path": part["crop_path"],
                            "score": part["score"],
                            "bbox": part["bbox"],
                        }
                        for part in normalized["parts"]
                    ],
                }
                if normalized.get("bbox") is not None:
                    record["bbox"] = normalized["bbox"]
                records.append(record)
            logger.info(
                f"Embed: {len(records)} linked instance(s) from "
                f"{manifest['images_processed']} image(s)"
            )
        else:
            crop_paths = sorted(
                p for p in input_dir.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES
            )
            if not crop_paths:
                logger.error(
                    f"No crop images (.jpg/.jpeg/.png) and no extract_manifest.json "
                    f"found in {input_dir}"
                )
                sys.exit(1)
            for crop_path in crop_paths:
                embeddings.append(_embed_single_normalized(embedder, crop_path))
                records.append(
                    {
                        "instance_id": crop_path.stem,
                        "image": str(crop_path),
                        "page": str(crop_path),
                        "panel": str(crop_path),
                        "crop_paths": [str(crop_path)],
                        "parts": [],
                    }
                )
            logger.info(f"Embed: {len(records)} single-part instance(s) from plain crops dir")
    except Exception as err:
        logger.error(f"Stage 2 embedding failed: {err}")
        sys.exit(1)

    if not records:
        logger.error(
            f"link_part_detections produced no instances from "
            f"{input_dir}/extract_manifest.json"
        )
        sys.exit(1)

    matrix = np.vstack(embeddings)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "embeddings.npy"
    np.save(embeddings_path, matrix.astype(np.float32))
    records_path = output_dir / "records.json"
    records_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    # 'match' must embed seeds in the SAME space as 'embed' produced. The
    # provenance below pins the embedding model for that stage (M3).
    meta_path = output_dir / "embed_meta.json"
    meta_path.write_text(
        # Absolute path: a later `catnip match` from a different cwd must
        # still resolve the same embedding model (final review M3 nit).
        json.dumps({"model": str(Path(model_path).resolve()), "instances": len(records)}, indent=2),
        encoding="utf-8",
    )
    logger.info(
        f"Embed complete: {matrix.shape[0]} instance(s) x {matrix.shape[1]} dims "
        f"-> {embeddings_path} + {records_path}"
    )
    return embeddings_path, records_path


def cmd_embed(args):
    """Stage 2: Compute embeddings for crops or a extract_manifest.json."""
    settings = _apply_overrides(args.config, args.override)
    _setup_logging(args.verbose)

    output_dir = Path(args.output) if args.output else Path(settings.paths.stage2_dir) / "embeddings"
    model_path = _resolve_stage2_checkpoint(settings, args.model)
    _run_embed(settings, Path(args.input_dir), output_dir, model_path)


# ---------------------------------------------------------------------------
# Stage 2: match (prototype bank -> two-tier decisions)
# ---------------------------------------------------------------------------


def _run_match(settings: Settings, embeddings_path: Path, output_dir: Path, threshold) -> tuple[Path, Path]:
    """Fit the PrototypeBank on Label Studio seeds and assign two-tier instances."""
    logger.info("Match: multi-prototype Izutsumi matching (Stage 2)")

    embeddings_path = Path(embeddings_path)
    if not embeddings_path.is_file():
        logger.error(f"Embeddings file not found: {embeddings_path} (run 'catnip embed' first)")
        sys.exit(1)
    records_path = embeddings_path.parent / "records.json"
    if not records_path.is_file():
        logger.error(
            f"records.json not found next to embeddings at {records_path}: "
            "'catnip match' requires the records written by 'catnip embed'"
        )
        sys.exit(1)

    logger.info(f"  Embeddings: {embeddings_path}")
    logger.info(f"  Output:     {output_dir}")

    embeddings = np.load(embeddings_path)
    if embeddings.ndim != 2:
        logger.error(f"Embeddings must be a 2-D [n, dim] array, got shape {embeddings.shape}")
        sys.exit(1)
    records = json.loads(records_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != embeddings.shape[0]:
        logger.error(
            f"records.json holds {len(records) if isinstance(records, list) else '?'} "
            f"entry(s) but embeddings.npy holds {embeddings.shape[0]} row(s)"
        )
        sys.exit(1)
    # Seeds must live in the embedding space 'embed' produced; prefer the
    # persisted provenance over config candidate resolution (M3).
    meta_path = embeddings_path.parent / "embed_meta.json"
    embed_model_path: Path | None = None
    if meta_path.is_file():
        try:
            embed_model_path = Path(json.loads(meta_path.read_text(encoding="utf-8"))["model"])
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning(
                f"Unreadable embed provenance at {meta_path} — falling back to "
                "config checkpoint resolution"
            )
    if embed_model_path is None:
        logger.warning(
            f"No embed_meta.json next to {embeddings_path} — resolving the "
            "Stage 2 checkpoint from config; match against embeddings produced "
            "by a different model will silently mis-score"
        )

    params = settings.params.clustering
    if threshold is not None:
        # --threshold overrides tau_core; the boundary band lower edge is
        # clamped along so the [tau_boundary, tau_core) band stays valid.
        params = params.model_copy(
            update={
                "tau_core": float(threshold),
                "tau_boundary": min(params.tau_boundary, float(threshold)),
            }
        )
        logger.info(
            f"  Threshold override: tau_core={params.tau_core}, "
            f"tau_boundary={params.tau_boundary}"
        )

    try:
        from src.reid.clustering import PrototypeBank, assign_izutsumi, build_review_gallery
        from src.reid.model import load_embedder

        embedder = load_embedder(
            str(embed_model_path) if embed_model_path is not None
            else _resolve_stage2_checkpoint(settings, None)
        )
        if embed_model_path is not None:
            logger.info(f"  Model: {embed_model_path} (from embed provenance)")
        bank = PrototypeBank(
            k_neighbors=params.k_neighbors, min_cluster_size=params.min_cluster_size
        )
        positive_seeds = _seed_images(
            settings.paths.izutsumi_dir, "Izutsumi must-link", embedder, bank.add_must_link
        )
        negative_seeds = _seed_images(
            settings.paths.not_izutsumi_dir, "cannot-link", embedder, bank.add_cannot_link
        )
        bank.fit_subclusters()

        # Records -> the assign_izutsumi contract: without panel segmentation
        # page == panel == the record's source image (documented default).
        assign_records = []
        for row, record in enumerate(records):
            image = str(record.get("image") or record["instance_id"])
            page = str(record.get("page") or image)
            panel = str(record.get("panel") or page)
            crop_paths = record.get("crop_paths") or []
            assign_record = {
                "instance_id": str(record["instance_id"]),
                "page": page,
                "panel": panel,
                "embedding": embeddings[row],
                "crop_path": str(crop_paths[0]) if crop_paths else image,
            }
            if record.get("bbox") is not None:
                # Instance footprint: empowers the Same-Panel Cannot-Link
                # spatial-distinctness exemption (heavy overlap = one
                # physical character chain, not two Izutsumi candidates).
                assign_record["bbox"] = [float(c) for c in record["bbox"]]
            assign_records.append(assign_record)
        decisions = assign_izutsumi(assign_records, bank, params)
    except SystemExit:
        raise
    except Exception as err:
        logger.error(f"Stage 2 matching failed: {err}")
        sys.exit(1)

    logger.info(
        f"Match: {len(bank.prototypes)} prototype(s) from {positive_seeds} must-link "
        f"seed(s) vs {negative_seeds} cannot-link seed(s)"
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = output_dir / "decisions.json"
    decisions_path.write_text(json.dumps(decisions, indent=2), encoding="utf-8")

    tier_counts = Counter(decision["tier"] for decision in decisions)
    logger.info(
        f"Match complete: {tier_counts.get('core', 0)} core / "
        f"{tier_counts.get('boundary', 0)} boundary / {tier_counts.get('reject', 0)} "
        f"reject -> {decisions_path}"
    )

    crops_by_instance = {
        decision["instance_id"]: decision["crop_path"]
        for decision in decisions
        if decision["tier"] == "boundary"
    }
    gallery_path = build_review_gallery(decisions, crops_by_instance, output_dir / "boundary_review.html")
    return decisions_path, gallery_path


def cmd_match(args):
    """Match embeddings against Izutsumi prototypes (two-tier output)."""
    settings = _apply_overrides(args.config, args.override)
    _setup_logging(args.verbose)

    output_dir = Path(args.output) if args.output else Path(settings.paths.output_dir) / "match"
    _run_match(settings, Path(args.embeddings), output_dir, args.threshold)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def cmd_pipeline(args):
    """Full pipeline: extract -> embed -> match (in-process, no subprocess)."""
    settings = _apply_overrides(args.config, args.override)
    _setup_logging(args.verbose)

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_dir) if args.output_dir else Path(settings.paths.output_dir)
    logger.info("Pipeline: extract -> embed -> match")
    logger.info(f"  Input:  {input_dir}")
    logger.info(f"  Output: {output_root}")

    logger.info("Pipeline stage 1/3: extract")
    stage1_model = _resolve_stage1_checkpoint(settings, None)
    _run_extract(settings, input_dir, output_root / "extract", stage1_model)

    logger.info("Pipeline stage 2/3: embed")
    stage2_model = _resolve_stage2_checkpoint(settings, None)
    _run_embed(settings, output_root / "extract", output_root / "embed", stage2_model)

    logger.info("Pipeline stage 3/3: match")
    _run_match(settings, output_root / "embed" / "embeddings.npy", output_root / "match", None)

    decisions_path = output_root / "match" / "decisions.json"
    logger.info(f"Pipeline complete: decisions at {decisions_path}")


def cmd_train_stage1(args):
    """Train Stage 1 YOLO26 model."""
    settings = _apply_overrides(args.config, args.override)
    _setup_logging(args.verbose)

    logger.info("Train Stage 1: YOLO26 face/body detection")
    logger.info(f"  Model: {settings.training.stage1.model}")
    logger.info(f"  IMGSZ: {settings.training.stage1.imgsz}")
    logger.info(f"  Epochs: {settings.training.stage1.epochs}")
    logger.info(f"  Batch: {settings.training.stage1.batch}")
    logger.info(f"  Device: {settings.training.stage1.device}")

    import subprocess
    script = Path(__file__).parent.parent / "scripts" / "train" / "stage1.py"
    cmd = [sys.executable, str(script)]
    if args.override:
        for ov in args.override:
            cmd.extend(["--override", ov])
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Train Stage 1 script failed with exit code {e.returncode}")
        sys.exit(e.returncode)


def cmd_train_stage2(args):
    """Train Stage 2 ReID model."""
    settings = _apply_overrides(args.config, args.override)
    _setup_logging(args.verbose)

    logger.info("Train Stage 2: SupCon Character Embedding (semi-supervised clustering)")
    logger.info(f"  Model: {settings.training.stage2.model}")
    logger.info(f"  Batch Size: {settings.training.stage2.batch_size}")
    logger.info(f"  Epochs: {settings.training.stage2.epochs}")
    logger.info(f"  Learning Rate: {settings.training.stage2.learning_rate}")
    logger.info(f"  Device: {settings.training.stage2.device}")

    import subprocess
    script = Path(__file__).parent.parent / "scripts" / "train" / "stage2.py"
    cmd = [sys.executable, str(script)]
    if args.override:
        for ov in args.override:
            cmd.extend(["--override", ov])
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Train Stage 2 script failed with exit code {e.returncode}")
        sys.exit(e.returncode)


def main():
    parser = argparse.ArgumentParser(
        prog="catnip",
        description="Character Re-Identification in manga — detect and identify Izutsumi.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", dest="global_verbose",
                        help="Enable DEBUG logging (global; merged with subcommand -v)")
    parser.add_argument("--config", type=Path, default="config/pipeline.yaml", help="Path to config file")
    # Global overrides live on a distinct dest: subparser namespaces start
    # empty, so a subcommand providing its own --override would otherwise
    # clobber values given before the subcommand (verified empirically).
    parser.add_argument("--override", action="append", default=[], dest="global_override",
                        metavar="KEY=VALUE",
                        help="Global override (merged with any subcommand --override)")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # extract
    p_extract = subparsers.add_parser("extract", help="Stage 1 SAHI face/body detection")
    p_extract.add_argument("--input-dir", type=Path, required=True, help="Directory of manga page images")
    p_extract.add_argument("--output-dir", type=Path, help="Output directory (writes crops/ + extract_manifest.json; default: from config)")
    p_extract.add_argument("--model", type=Path, help="Path to YOLO26 model (default: candidate resolution)")
    p_extract.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging")
    p_extract.add_argument("--override", action="append", default=argparse.SUPPRESS, metavar="KEY=VALUE")
    p_extract.set_defaults(func=cmd_extract)

    # embed
    p_embed = subparsers.add_parser("embed", help="Stage 2 ReID embedding computation")
    p_embed.add_argument("--input-dir", type=Path, required=True,
                         help="Directory of character crops, or a directory holding extract_manifest.json")
    p_embed.add_argument("--output", type=Path, help="Output directory (writes embeddings.npy + records.json)")
    p_embed.add_argument("--model", type=Path, help="Path to ReID model (default: candidate resolution)")
    p_embed.add_argument("--verbose", "-v", action="store_true")
    p_embed.add_argument("--override", action="append", default=argparse.SUPPRESS, metavar="KEY=VALUE")
    p_embed.set_defaults(func=cmd_embed)

    # match
    p_match = subparsers.add_parser("match", help="Match embeddings against Izutsumi prototypes")
    p_match.add_argument("--embeddings", type=Path, required=True, help="Path to embeddings .npy file")
    p_match.add_argument("--threshold", type=float, default=None,
                         help="Optional override for params.clustering.tau_core (default: config value)")
    p_match.add_argument("--output", type=Path, help="Output directory (writes decisions.json + boundary_review.html)")
    p_match.add_argument("--verbose", "-v", action="store_true")
    p_match.add_argument("--override", action="append", default=argparse.SUPPRESS, metavar="KEY=VALUE")
    p_match.set_defaults(func=cmd_match)

    # pipeline
    p_pipeline = subparsers.add_parser("pipeline", help="Full pipeline: extract -> embed -> match")
    p_pipeline.add_argument("--input-dir", type=Path, required=True, help="Directory of manga page images")
    p_pipeline.add_argument("--output-dir", type=Path, help="Output directory (writes extract/, embed/, match/)")
    p_pipeline.add_argument("--verbose", "-v", action="store_true")
    p_pipeline.add_argument("--override", action="append", default=argparse.SUPPRESS, metavar="KEY=VALUE")
    p_pipeline.set_defaults(func=cmd_pipeline)

    # train stage1
    p_train1 = subparsers.add_parser("train", help="Training commands")
    train_subs = p_train1.add_subparsers(dest="train_command")
    p_t1 = train_subs.add_parser("stage1", help="Train Stage 1 YOLO26 face/body detection")
    p_t1.add_argument("--verbose", "-v", action="store_true")
    p_t1.add_argument("--override", action="append", default=argparse.SUPPRESS, metavar="KEY=VALUE")
    p_t1.set_defaults(func=cmd_train_stage1)

    p_t2 = train_subs.add_parser("stage2", help="Train Stage 2 SupCon character embeddings")
    p_t2.add_argument("--verbose", "-v", action="store_true")
    p_t2.add_argument("--override", action="append", default=argparse.SUPPRESS, metavar="KEY=VALUE")
    p_t2.set_defaults(func=cmd_train_stage2)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Merge chain (argparse sub-namespaces start empty, so sub-provided
    # flags overwrite attrs copied back onto the main namespace): globals
    # first, then the subcommand's own --override entries. _apply_overrides
    # applies entries in order and later entries overwrite earlier keys, so
    # a subcommand flag deliberately beats the same key given globally.
    global_overrides = list(getattr(args, "global_override", None) or [])
    sub_overrides = list(getattr(args, "override", None) or [])
    args.override = global_overrides + sub_overrides
    args.verbose = bool(getattr(args, "verbose", False)) or bool(
        getattr(args, "global_verbose", False)
    )

    args.func(args)


if __name__ == "__main__":
    main()
