"""Unit tests for the Catnip CLI extract/embed/match/pipeline commands.

All heavy seams (SahiExtractor, load_embedder, fuse_part_embeddings) are
monkeypatched fakes — no real model weights, no network, no dataset access.
link_part_detections is exercised REAL against synthetic manifests, and the
fake embedder derives embeddings from IMAGE CONTENT, matching the real
contract (CharacterEmbedder.embed_single takes an RGB uint8 array, not a
path). Every output and seed directory is overridden to a tmp directory.
"""

import json
import logging
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

import src.cli as cli

CONFIG = Path("config/pipeline.yaml")

# Deterministic embedding geometry: izutsumi seeds/instances -> +e0,
# negatives (marcille / not_izutsumi) -> +e1.
DIM = 8
_POS = np.zeros(DIM); _POS[0] = 1.0
_NEG = np.zeros(DIM); _NEG[1] = 1.0

# Nested anatomy per image: face is inside head, head inside body (IoS 1.0),
# so the real link_part_detections emits exactly one instance per image.
_NESTED_BOXES = {
    "body": [0, 0, 100, 100],
    "head": [10, 10, 70, 70],
    "face": [20, 20, 50, 50],
}


# ---------------------------------------------------------------------------
# Fake seams
# ---------------------------------------------------------------------------


def fake_crop_kind(stem: str) -> str:
    """Encoding rule shared by test images: 'neg' vs 'pos' by filename."""
    if "not_izutsumi" in stem or "marcille" in stem:
        return "neg"
    return "pos"


def write_crop_image(path: Path, kind: str) -> None:
    """Write a real 8x8 cv2 image: red (BGR 0,0,255) for pos, blue for neg."""
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    if kind == "pos":
        image[..., 2] = 255  # red
    else:
        image[..., 0] = 255  # blue
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


class FakeEmbedder:
    def __init__(self, model_path=None):
        self.model_path = model_path

    def embed_single(self, image_rgb: np.ndarray) -> np.ndarray:
        """Content-based fake: red-dominant -> +e0, blue-dominant -> +e1."""
        red = image_rgb[..., 0].mean()
        blue = image_rgb[..., 2].mean()
        if red > blue and red > 100:
            return _POS.copy()
        if blue > red and blue > 100:
            return _NEG.copy()
        raise AssertionError(
            f"test crop neither red nor blue (red={red:.1f}, blue={blue:.1f})"
        )


def fake_fuse_part_embeddings(parts, scores, weights) -> np.ndarray:
    """Plan-faithful fusion: unit vector of weighted, confidence-scaled parts."""
    names = sorted(parts)
    fused = np.zeros(DIM)
    for name in names:
        fused = fused + float(weights[name]) * float(scores[name]) * np.asarray(parts[name])
    return (fused / np.linalg.norm(fused)).astype(np.float32)


class FakeSahiExtractor:
    last_call: dict = {}

    def __init__(self, model_path=None, device=None):
        FakeSahiExtractor.last_call = {"model_path": model_path, "device": device}

    def extract_from_directory(self, input_dir=None, output_dir=None, save_crops=True):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        crops = []
        images = sorted(
            p for p in Path(input_dir).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        for image in images:
            for class_name, bbox in _NESTED_BOXES.items():
                crop_path = output_dir / f"{image.stem}_crop_{class_name}.jpg"
                write_crop_image(crop_path, fake_crop_kind(image.stem))
                crops.append(
                    {
                        "image": str(image),
                        "crop_path": str(crop_path),
                        "bbox": list(bbox),
                        "class_name": class_name,
                        "score": 0.9,
                    }
                )
        FakeSahiExtractor.last_call.update(
            {"output_dir": output_dir, "save_crops": save_crops, "input_dir": str(input_dir)}
        )
        return types.SimpleNamespace(
            images_processed=len(images), crops_saved=len(crops), crops=crops
        )


def install_fake_reid_modules(monkeypatch) -> None:
    """Patch the heavy seams; link_part_detections stays REAL (it is pure)."""
    monkeypatch.setattr(
        "src.detection.sahi_extractor.SahiExtractor", FakeSahiExtractor
    )
    install_fake_module(
        monkeypatch,
        "src.reid.model",
        load_embedder=lambda model_path: FakeEmbedder(model_path),
        fuse_part_embeddings=fake_fuse_part_embeddings,
    )


def install_fake_module(monkeypatch, name: str, **attributes) -> types.ModuleType:
    """Patch an existing module or inject a brand-new one into sys.modules."""
    try:
        module = __import__(name, fromlist=["_"])
    except ImportError:
        module = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
    for attribute, value in attributes.items():
        monkeypatch.setattr(module, attribute, value)
    return module


def run_cmd(func, **kwargs):
    kwargs.setdefault("override", [])
    args = types.SimpleNamespace(config=CONFIG, verbose=False, **kwargs)
    func(args)


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------


def test_cmd_extract_writes_manifest_and_uses_fake_extractor(
    monkeypatch, tmp_path
):
    install_fake_reid_modules(monkeypatch)
    monkeypatch.setattr(
        cli, "_resolve_stage1_checkpoint", lambda settings, model: tmp_path / "fake-stage1.pt"
    )
    input_dir = tmp_path / "pages"
    input_dir.mkdir()
    (input_dir / "page_a.jpg").write_bytes(b"a")
    (input_dir / "page_b.jpg").write_bytes(b"b")

    run_cmd(
        cli.cmd_extract,
        input_dir=input_dir,
        output_dir=tmp_path / "out",
        model=None,
    )

    manifest_path = tmp_path / "out" / "extract_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["images_processed"] == 2
    assert manifest["crops_saved"] == 6
    assert len(manifest["crops"]) == 6
    assert {crop["class_name"] for crop in manifest["crops"]} == {"body", "head", "face"}
    # the fake extractor really wrote its crops under <output>/crops
    assert FakeSahiExtractor.last_call["save_crops"] is True
    assert FakeSahiExtractor.last_call["output_dir"] == tmp_path / "out" / "crops"
    assert (tmp_path / "out" / "crops" / "page_a_crop_body.jpg").is_file()


def test_cmd_extract_missing_checkpoint_exits_with_candidates(
    monkeypatch, tmp_path, caplog
):
    install_fake_reid_modules(monkeypatch)
    # explicit --model that does not exist
    with pytest.raises(SystemExit) as missing_explicit:
        run_cmd(
            cli.cmd_extract,
            input_dir=tmp_path,
            output_dir=tmp_path / "out",
            model=tmp_path / "definitely-missing.pt",
        )
    assert missing_explicit.value.code == 1
    assert any("not found" in r.message for r in caplog.records)

    # and default resolution with no candidate on disk anywhere
    monkeypatch.setattr(
        cli, "_stage1_checkpoint_candidates", lambda settings: [tmp_path / "absent.pt"]
    )
    caplog.clear()
    with pytest.raises(SystemExit) as missing_resolved:
        run_cmd(
            cli.cmd_extract,
            input_dir=tmp_path,
            output_dir=tmp_path / "out",
            model=None,
        )
    assert missing_resolved.value.code == 1
    assert any("absent.pt" in r.message for r in caplog.records)


def test_cmd_extract_missing_input_dir_exits(monkeypatch, tmp_path):
    install_fake_reid_modules(monkeypatch)
    monkeypatch.setattr(
        cli, "_resolve_stage1_checkpoint", lambda settings, model: tmp_path / "fake-stage1.pt"
    )
    with pytest.raises(SystemExit) as exc:
        run_cmd(
            cli.cmd_extract,
            input_dir=tmp_path / "nonexistent",
            output_dir=tmp_path / "out",
            model=None,
        )
    assert exc.value.code == 1


# ---------------------------------------------------------------------------
# embed
# ---------------------------------------------------------------------------


def _write_manifest(tmp_path, broken_crop: bool = False) -> Path:
    """Synthetic extract_manifest.json with two nested-instance images."""
    input_dir = tmp_path / "in"
    input_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = input_dir / "crops"
    crops = []
    for image_name in ("izutsumi_p1.jpg", "marcille_p2.jpg"):
        stem = Path(image_name).stem
        (input_dir / image_name).write_bytes(b"page")
        for class_name, bbox in _NESTED_BOXES.items():
            crop_path = crops_dir / f"{stem}_crop_{class_name}.jpg"
            if not (broken_crop and image_name == "marcille_p2.jpg" and class_name == "face"):
                write_crop_image(crop_path, fake_crop_kind(stem))
            else:
                crop_path = crops_dir / "nowhere_face.jpg"  # deliberately missing
            crops.append(
                {
                    "image": str(input_dir / image_name),
                    "crop_path": str(crop_path),
                    "bbox": list(bbox),
                    "class_name": class_name,
                    "score": 0.9,
                }
            )
    manifest = {"images_processed": 2, "crops_saved": len(crops) - 1, "crops": crops}
    (input_dir / "extract_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return input_dir


def test_cmd_embed_manifest_mode_writes_consistent_records_and_embeddings(
    monkeypatch, tmp_path
):
    install_fake_reid_modules(monkeypatch)
    monkeypatch.setattr(
        cli, "_resolve_stage2_checkpoint", lambda settings, model: tmp_path / "fake-reid.pt"
    )
    input_dir = _write_manifest(tmp_path)

    run_cmd(cli.cmd_embed, input_dir=input_dir, output=tmp_path / "out", model=None)

    records = json.loads((tmp_path / "out" / "records.json").read_text(encoding="utf-8"))
    embeddings = np.load(tmp_path / "out" / "embeddings.npy")

    assert embeddings.shape == (2, DIM)
    assert embeddings.dtype == np.float32
    assert len(records) == 2
    by_id = {record["instance_id"]: record for record in records}
    # real link_part_detections ids: "<image>#<001-based idx>" with the
    # full manifest image path
    assert set(by_id) == {
        f"{input_dir}/izutsumi_p1.jpg#001",
        f"{input_dir}/marcille_p2.jpg#001",
    }
    for record in records:
        # no panel metadata on CharacterInstance: page == panel == source image
        assert record["page"] == record["panel"] == record["image"]
        assert [p["class_name"] for p in record["parts"]] == ["body", "head", "face"]
        assert len(record["crop_paths"]) == 3
        assert all(Path(p).is_file() for p in record["crop_paths"])
        # part bboxes flow through records (panel distinctness + provenance):
        # the instance box is the outermost part (body).
        assert all(len(p["bbox"]) == 4 for p in record["parts"])
        assert record["bbox"] == next(p["bbox"] for p in record["parts"] if p["class_name"] == "body")

    # match must reuse the exact embedding model: 'embed' persists provenance.
    meta = json.loads((tmp_path / "out" / "embed_meta.json").read_text(encoding="utf-8"))
    assert meta["model"].endswith("fake-reid.pt")
    assert meta["instances"] == 2

    # embeddings.npy rows equal the fusion of the CONTENT-derived fake parts:
    # izutsumi parts are red (+e0), marcille parts are blue (+e1)
    assert np.allclose(embeddings[0], _POS.astype(np.float32), atol=1e-6)
    assert np.allclose(embeddings[1], _NEG.astype(np.float32), atol=1e-6)
    assert np.allclose(np.linalg.norm(embeddings.astype(np.float64), axis=1), 1.0, atol=1e-6)


def test_cmd_embed_flat_crops_dir_makes_single_part_instances(monkeypatch, tmp_path):
    install_fake_reid_modules(monkeypatch)
    monkeypatch.setattr(
        cli, "_resolve_stage2_checkpoint", lambda settings, model: tmp_path / "fake-reid.pt"
    )
    input_dir = tmp_path / "crops"
    input_dir.mkdir()
    write_crop_image(input_dir / "a.jpg", "pos")
    write_crop_image(input_dir / "b.png", "pos")

    run_cmd(cli.cmd_embed, input_dir=input_dir, output=tmp_path / "out", model=None)

    records = json.loads((tmp_path / "out" / "records.json").read_text(encoding="utf-8"))
    embeddings = np.load(tmp_path / "out" / "embeddings.npy")
    assert [r["instance_id"] for r in records] == ["a", "b"]
    assert all(r["parts"] == [] and len(r["crop_paths"]) == 1 for r in records)
    assert all(r["page"] == r["panel"] == r["image"] for r in records)
    assert embeddings.shape == (2, DIM)
    assert np.allclose(embeddings.astype(np.float64), np.stack([_POS, _POS]), atol=1e-6)


def test_cmd_embed_missing_crop_file_is_invalid_state(monkeypatch, tmp_path, caplog):
    install_fake_reid_modules(monkeypatch)
    monkeypatch.setattr(
        cli, "_resolve_stage2_checkpoint", lambda settings, model: tmp_path / "fake-reid.pt"
    )
    input_dir = _write_manifest(tmp_path, broken_crop=True)

    with pytest.raises(SystemExit) as exc:
        run_cmd(cli.cmd_embed, input_dir=input_dir, output=tmp_path / "out", model=None)
    assert exc.value.code == 1
    assert any("missing" in r.message and "face" in r.message for r in caplog.records)


def test_cmd_embed_missing_model_exits(monkeypatch, tmp_path, caplog):
    install_fake_reid_modules(monkeypatch)
    input_dir = tmp_path / "crops"
    input_dir.mkdir()
    with pytest.raises(SystemExit) as exc:
        run_cmd(
            cli.cmd_embed,
            input_dir=input_dir,
            output=tmp_path / "out",
            model=tmp_path / "no-such-embedder.pt",
        )
    assert exc.value.code == 1
    assert any("not found" in r.message for r in caplog.records)


def test_cmd_embed_empty_input_dir_exits(monkeypatch, tmp_path, caplog):
    install_fake_reid_modules(monkeypatch)
    monkeypatch.setattr(
        cli, "_resolve_stage2_checkpoint", lambda settings, model: tmp_path / "fake-reid.pt"
    )
    input_dir = tmp_path / "empty"
    input_dir.mkdir()
    with pytest.raises(SystemExit) as exc:
        run_cmd(cli.cmd_embed, input_dir=input_dir, output=tmp_path / "out", model=None)
    assert exc.value.code == 1
    assert any("No crop images" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# match
# ---------------------------------------------------------------------------


def _write_match_inputs(tmp_path):
    """embeddings.npy + records.json: core / boundary / rejected geometry."""
    records = [
        {"instance_id": "iz-core", "image": "p1.jpg", "page": "p1.jpg", "panel": "p1.jpg",
         "crop_paths": ["crops/iz-core.jpg"], "parts": []},
        {"instance_id": "iz-back", "image": "p2.jpg", "page": "p2.jpg", "panel": "p2.jpg",
         "crop_paths": ["crops/iz-back.jpg"], "parts": []},
        {"instance_id": "marcille", "image": "p2b.jpg", "page": "p2b.jpg", "panel": "p2b.jpg",
         "crop_paths": ["crops/marcille.jpg"], "parts": []},
    ]
    mid = 0.75 * _POS + np.sqrt(1 - 0.75**2) * np.array([0, 0, 1, 0, 0, 0, 0, 0])
    embeddings = np.stack([_POS, mid, _NEG]).astype(np.float32)
    embed_dir = tmp_path / "embed"
    embed_dir.mkdir(parents=True)
    np.save(embed_dir / "embeddings.npy", embeddings)
    (embed_dir / "records.json").write_text(json.dumps(records), encoding="utf-8")
    return embed_dir


def _seed_overrides(tmp_path) -> list[str]:
    iz = tmp_path / "seeds" / "izutsumi"
    neg = tmp_path / "seeds" / "not_izutsumi"
    iz.mkdir(parents=True, exist_ok=True)
    neg.mkdir(parents=True, exist_ok=True)
    write_crop_image(iz / "izutsumi_0.jpg", "pos")
    write_crop_image(iz / "izutsumi_1.jpg", "pos")
    write_crop_image(neg / "not_izutsumi_0.jpg", "neg")
    write_crop_image(neg / "not_izutsumi_1.jpg", "neg")
    return [
        f"paths.izutsumi_dir={iz}",
        f"paths.not_izutsumi_dir={neg}",
    ]


def test_cmd_match_two_tier_decisions_and_review_gallery(monkeypatch, tmp_path):
    install_fake_reid_modules(monkeypatch)
    monkeypatch.setattr(
        cli, "_resolve_stage2_checkpoint", lambda settings, model: tmp_path / "fake-reid.pt"
    )
    embed_dir = _write_match_inputs(tmp_path)

    run_cmd(
        cli.cmd_match,
        embeddings=embed_dir / "embeddings.npy",
        output=tmp_path / "out",
        threshold=None,
        override=_seed_overrides(tmp_path),
    )

    decisions = json.loads((tmp_path / "out" / "decisions.json").read_text(encoding="utf-8"))
    by_id = {d["instance_id"]: d for d in decisions}
    assert by_id["iz-core"]["tier"] == "core"
    assert by_id["iz-back"]["tier"] == "boundary"
    assert by_id["marcille"]["tier"] == "reject"
    assert by_id["marcille"]["score"] < 0  # veto applied: 2*s_pos - s_neg << 0
    assert all(d["reasons"] and "crop_path" in d for d in decisions)

    gallery = (tmp_path / "out" / "boundary_review.html").read_text(encoding="utf-8")
    assert "iz-back" in gallery
    assert "iz-core" not in gallery
    assert "marcille" not in gallery


def test_cmd_match_threshold_overrides_tau_core(monkeypatch, tmp_path):
    install_fake_reid_modules(monkeypatch)
    monkeypatch.setattr(
        cli, "_resolve_stage2_checkpoint", lambda settings, model: tmp_path / "fake-reid.pt"
    )
    embed_dir = _write_match_inputs(tmp_path)

    run_cmd(
        cli.cmd_match,
        embeddings=embed_dir / "embeddings.npy",
        output=tmp_path / "out",
        threshold=0.5,
        override=_seed_overrides(tmp_path),
    )
    decisions = json.loads((tmp_path / "out" / "decisions.json").read_text(encoding="utf-8"))
    by_id = {d["instance_id"]: d for d in decisions}
    # tau_core lowered to 0.5: the 0.75 boundary instance auto-tags as core
    assert by_id["iz-back"]["tier"] == "core"


def test_cmd_match_missing_records_json_exits(monkeypatch, tmp_path, caplog):
    install_fake_reid_modules(monkeypatch)
    embed_dir = tmp_path / "embed"
    embed_dir.mkdir(parents=True)
    np.save(embed_dir / "embeddings.npy", np.zeros((2, 4), dtype=np.float32))

    with pytest.raises(SystemExit) as exc:
        run_cmd(
            cli.cmd_match,
            embeddings=embed_dir / "embeddings.npy",
            output=tmp_path / "out",
            threshold=None,
        )
    assert exc.value.code == 1
    assert any("records.json not found" in r.message for r in caplog.records)


def test_cmd_match_row_count_mismatch_exits(monkeypatch, tmp_path, caplog):
    install_fake_reid_modules(monkeypatch)
    embed_dir = tmp_path / "embed"
    embed_dir.mkdir(parents=True)
    np.save(embed_dir / "embeddings.npy", np.zeros((3, 4), dtype=np.float32))
    (embed_dir / "records.json").write_text(
        json.dumps([{"instance_id": "only-one", "image": "x", "page": "x", "panel": "x",
                     "crop_paths": [], "parts": []}]),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc:
        run_cmd(
            cli.cmd_match,
            embeddings=embed_dir / "embeddings.npy",
            output=tmp_path / "out",
            threshold=None,
        )
    assert exc.value.code == 1
    assert any("1 entry" in r.message and "3 row" in r.message for r in caplog.records)


def test_cmd_match_missing_seed_dir_exits(monkeypatch, tmp_path, caplog):
    install_fake_reid_modules(monkeypatch)
    monkeypatch.setattr(
        cli, "_resolve_stage2_checkpoint", lambda settings, model: tmp_path / "fake-reid.pt"
    )
    embed_dir = _write_match_inputs(tmp_path)

    with pytest.raises(SystemExit) as exc:
        run_cmd(
            cli.cmd_match,
            embeddings=embed_dir / "embeddings.npy",
            output=tmp_path / "out",
            threshold=None,
            override=[f"paths.izutsumi_dir={tmp_path / 'no-such-seeds'}",
                      f"paths.not_izutsumi_dir={tmp_path / 'no-such-negatives'}"],
        )
    assert exc.value.code == 1
    assert any("seed directory not found" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------


def test_cmd_pipeline_chains_all_stages(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="catnip")
    install_fake_reid_modules(monkeypatch)
    monkeypatch.setattr(
        cli, "_resolve_stage1_checkpoint", lambda settings, model: tmp_path / "fake-stage1.pt"
    )
    monkeypatch.setattr(
        cli, "_resolve_stage2_checkpoint", lambda settings, model: tmp_path / "fake-reid.pt"
    )
    input_dir = tmp_path / "pages"
    input_dir.mkdir()
    write_crop_image(input_dir / "izutsumi_page.jpg", "pos")
    write_crop_image(input_dir / "marcille_page.jpg", "neg")

    run_cmd(
        cli.cmd_pipeline,
        input_dir=input_dir,
        output_dir=tmp_path / "pipeline-out",
        override=_seed_overrides(tmp_path),
    )

    # stage 1 artifacts
    extract_manifest = tmp_path / "pipeline-out" / "extract" / "extract_manifest.json"
    manifest = json.loads(extract_manifest.read_text(encoding="utf-8"))
    assert manifest["images_processed"] == 2
    assert (tmp_path / "pipeline-out" / "extract" / "crops" / "izutsumi_page_crop_body.jpg").is_file()

    # stage 2 artifacts: real linker gives one nested instance per image
    embeddings = np.load(tmp_path / "pipeline-out" / "embed" / "embeddings.npy")
    records = json.loads(
        (tmp_path / "pipeline-out" / "embed" / "records.json").read_text(encoding="utf-8")
    )
    assert embeddings.shape == (2, DIM)
    assert [r["instance_id"] for r in records] == [
        f"{input_dir}/izutsumi_page.jpg#001",
        f"{input_dir}/marcille_page.jpg#001",
    ]

    # stage 3 artifacts: izutsumi page instance auto-tags core, marcille rejected
    decisions = json.loads(
        (tmp_path / "pipeline-out" / "match" / "decisions.json").read_text(encoding="utf-8")
    )
    by_id = {d["instance_id"]: d for d in decisions}
    assert by_id[f"{input_dir}/izutsumi_page.jpg#001"]["tier"] == "core"
    assert by_id[f"{input_dir}/marcille_page.jpg#001"]["tier"] == "reject"
    assert (tmp_path / "pipeline-out" / "match" / "boundary_review.html").is_file()

    # per-stage log lines
    messages = [record.message for record in caplog.records]
    assert any("Pipeline stage 1/3: extract" in m for m in messages)
    assert any("Pipeline stage 2/3: embed" in m for m in messages)
    assert any("Pipeline stage 3/3: match" in m for m in messages)


def test_cmd_pipeline_missing_stage1_checkpoint_exits_midway(
    monkeypatch, tmp_path, caplog
):
    install_fake_reid_modules(monkeypatch)
    monkeypatch.setattr(
        cli, "_stage1_checkpoint_candidates", lambda settings: [tmp_path / "absent.pt"]
    )
    input_dir = tmp_path / "pages"
    input_dir.mkdir()

    with pytest.raises(SystemExit) as exc:
        run_cmd(cli.cmd_pipeline, input_dir=input_dir, output_dir=tmp_path / "out")
    assert exc.value.code == 1
    # failed at stage 1: no embed/ or match/ artifacts
    assert not (tmp_path / "out" / "embed").exists()
    assert any("absent.pt" in r.message for r in caplog.records)


def test_cli_merges_global_and_subcommand_overrides(monkeypatch) -> None:
    """Global --override survives a subcommand that passes its own --override.

    argparse sub-namespaces start empty; without the merge in main() the
    sub-side list silently drops global overrides (final review M10).
    Globals are prepended (applied FIRST) so a same-key subcommand entry
    applied later overwrites the global one (final review C2).
    """
    from src import cli as cli_module

    seen = {}
    original = cli_module._apply_overrides

    def spy(config_path, overrides):
        seen["overrides"] = list(overrides)
        # Do not actually load or validate settings here; the merge order
        # is the object under test.
        raise SystemExit(1)

    monkeypatch.setattr(cli_module, "_apply_overrides", spy)
    monkeypatch.setattr(
        sys, "argv",
        [
            "catnip", "--override", "params.device=cpu",
            "pipeline",
            "--input-dir", "definitely-missing",
            "--override", "training.stage1.epochs=9",
        ],
    )

    with pytest.raises(SystemExit):
        cli_module.main()

    assert seen["overrides"] == ["params.device=cpu", "training.stage1.epochs=9"]


def test_cli_subcommand_override_wins_same_key(monkeypatch) -> None:
    """A same-key subcommand --override overrides the global value (C1 order).

    _apply_overrides applies entries in list order with later keys winning,
    so the merged list must end with the subcommand's entry.
    """
    from src import cli as cli_module

    settings = cli_module._apply_overrides(
        Path("config/pipeline.yaml"),
        ["training.stage1.epochs=75", "training.stage1.epochs=9"],
    )
    assert settings.training.stage1.epochs == 9
