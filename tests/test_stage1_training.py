"""Unit tests for the Stage 1 v2 training profile (scripts/train/stage1.py).

Synthetic and CPU-only by design: no catnip-data tree, no CUDA/MPS, no
network, and no ultralytics training instantiation. Covers:
(a) run/artifact naming (historical vs configured vs polish),
(b) the OOM batch ladder,
(c) the RandomGrayscale pure op (determinism, p semantics, jitter math),
(d) the build_transforms patch installer and its transform element,
(e) _build_polish_dataset on a synthetic sliced tree,
(f) --polish CLI guards (fail fast before any model load).
"""

import logging
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.train import stage1
from src.config import Stage1TrainingParams, get_settings
from ultralytics.data.augment import Compose


def _stage1_params(**updates) -> Stage1TrainingParams:
    """A copy of the real configured stage1 params with fields overridden."""
    return get_settings().training.stage1.model_copy(update=updates)


def _color_image() -> np.ndarray:
    """Bands with distinct BGR values so grayscale conversion is observable."""
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:, :, 0] = 250  # B
    img[:, :, 1] = 120  # G
    img[:, :, 2] = 30   # R
    return img


# ---------------------------------------------------------------------------
# (a) Naming: _resolve_run_names
# ---------------------------------------------------------------------------

def test_naming_historical_when_fields_empty_and_subset() -> None:
    params = _stage1_params(run_name="", target_model_name="", use_subset=True)
    assert stage1._resolve_run_names(params) == (
        "stage1_m3", "yolo26_stage1_body_head_face_m3.pt",
    )


def test_naming_historical_when_fields_empty_and_full_dataset() -> None:
    params = _stage1_params(run_name="", target_model_name="", use_subset=False)
    assert stage1._resolve_run_names(params) == (
        "stage1", "yolo26_stage1_body_head_face.pt",
    )


def test_naming_configured_fields_win_over_subset_heuristic() -> None:
    # Config carries the v2 names while use_subset mimics the m3 overrides:
    # explicit run_name/target still take precedence.
    params = _stage1_params(use_subset=True)
    assert stage1._resolve_run_names(params) == (
        "stage1_v2", "yolo26m_stage1_v2.pt",
    )


def test_naming_polish_suffix_reuses_target_name() -> None:
    run_name, target = stage1._resolve_run_names(_stage1_params(), polish=True)
    assert run_name == "stage1_v2_polish"
    assert target == "yolo26m_stage1_v2.pt"


def test_naming_polish_suffix_appends_to_historical_names() -> None:
    params = _stage1_params(run_name="", target_model_name="", use_subset=True)
    assert stage1._resolve_run_names(params, polish=True) == (
        "stage1_m3_polish", "yolo26_stage1_body_head_face_m3.pt",
    )


# ---------------------------------------------------------------------------
# (b) OOM batch ladder
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "batch,expected",
    [
        (8, [8, 4, 2]),                    # M3 subset profile
        (48, [48, 32, 16, 8, 4, 2]),       # L4 v2 profile
        (32, [32, 16, 8, 4, 2]),           # equal fallbacks never duplicated
        (2, [2]),
    ],
)
def test_oom_batch_ladder(batch: int, expected: list[int]) -> None:
    assert stage1._oom_batch_ladder(batch) == expected


@pytest.mark.parametrize("batch", [0, -4])
def test_oom_batch_ladder_rejects_nonpositive(batch: int) -> None:
    with pytest.raises(ValueError, match="batch"):
        stage1._oom_batch_ladder(batch)


# ---------------------------------------------------------------------------
# (c) RandomGrayscale pure op
# ---------------------------------------------------------------------------

def test_random_grayscale_p1_always_converts_to_3ch_gray() -> None:
    img = _color_image()
    out = stage1._apply_random_grayscale(img, 1.0, random.Random(0))
    assert out.shape == img.shape and out.dtype == img.dtype
    assert out.ndim == 3 and out.shape[-1] == 3
    np.testing.assert_array_equal(out[..., 0], out[..., 1])
    np.testing.assert_array_equal(out[..., 1], out[..., 2])
    assert not np.array_equal(out, img)  # colored input cannot survive as-is


def test_random_grayscale_p0_never_converts() -> None:
    img = _color_image()
    for seed in (0, 1, 2, 3):
        out = stage1._apply_random_grayscale(img, 0.0, random.Random(seed))
        np.testing.assert_array_equal(out, img)


def test_random_grayscale_deterministic_under_seeded_rng() -> None:
    img = _color_image()
    first = stage1._apply_random_grayscale(img, 1.0, random.Random(42))
    second = stage1._apply_random_grayscale(img, 1.0, random.Random(42))
    np.testing.assert_array_equal(first, second)


def test_random_grayscale_jitter_matches_documented_hyperparams() -> None:
    img = _color_image()
    out = stage1._apply_random_grayscale(img, 1.0, random.Random(7))
    # Replay the documented draw sequence: 1x bernoulli p, then alpha, beta.
    mirror = random.Random(7)
    mirror.random()  # p draw
    alpha = mirror.uniform(*stage1._GRAYSCALE_JITTER_ALPHA_RANGE)
    beta = mirror.uniform(*stage1._GRAYSCALE_JITTER_BETA_RANGE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    expected = np.clip(
        np.round(gray.astype(np.float32) * alpha + beta), 0, 255
    ).astype(np.uint8)
    np.testing.assert_array_equal(out, np.repeat(expected[..., None], 3, axis=-1))


def test_random_grayscale_leaves_non_3channel_untouched() -> None:
    gray_2d = np.full((4, 4), 128, dtype=np.uint8)
    assert stage1._apply_random_grayscale(gray_2d, 1.0, random.Random(0)) is gray_2d


# ---------------------------------------------------------------------------
# (d) Patch installer + transform element (no training instantiation)
# ---------------------------------------------------------------------------

def test_install_random_grayscale_patched_once_and_wraps_class() -> None:
    from ultralytics.data import dataset as dataset_module

    original = dataset_module.YOLODataset.__dict__["build_transforms"]
    flag_before = stage1._random_grayscale_patched
    try:
        stage1._random_grayscale_patched = False

        assert stage1._install_random_grayscale(0.25) is True
        wrapped = dataset_module.YOLODataset.__dict__["build_transforms"]
        assert wrapped is not original

        # Disabling or re-installing must not stack wrappers.
        assert stage1._install_random_grayscale(0.5) is False
        assert stage1._install_random_grayscale(0.0) is False
        assert dataset_module.YOLODataset.__dict__["build_transforms"] is wrapped
    finally:
        dataset_module.YOLODataset.build_transforms = original
        stage1._random_grayscale_patched = flag_before


def test_insert_random_grayscale_places_transform_before_format() -> None:
    head, mid, fmt = object(), object(), object()
    compose = Compose([head, mid, fmt])
    out = stage1._insert_random_grayscale(compose, p=0.25, augmenting=True)

    assert out is not compose          # new Compose; no hidden mutation
    assert isinstance(out.transforms[-2], stage1._RandomGrayscaleTransform)
    assert out.transforms[-1] is fmt   # Format stays last (tensor conversion)
    assert out.transforms[0] is head and out.transforms[1] is mid


def test_insert_random_grayscale_disabled_paths_return_input() -> None:
    compose = Compose([object(), object()])
    assert stage1._insert_random_grayscale(compose, p=0.0, augmenting=True) is compose
    # val/test datasets build with augment=False: must stay inference-identical
    assert stage1._insert_random_grayscale(compose, p=0.25, augmenting=False) is compose


def test_random_grayscale_transform_delegates_to_op_with_injected_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple] = []

    def spy(img, p, rng):
        calls.append((img, p, rng))
        return img

    monkeypatch.setattr(stage1, "_apply_random_grayscale", spy)
    img = _color_image()
    rng = random.Random(11)

    labels = stage1._RandomGrayscaleTransform(p=0.5, rng=rng)({"img": img})

    assert len(calls) == 1
    assert calls[0][0] is img
    assert calls[0][1] == 0.5
    assert calls[0][2] is rng
    assert labels["img"] is img


def test_random_grayscale_transform_skips_non_numpy_img(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        stage1, "_apply_random_grayscale",
        lambda *a: pytest.fail("grayscale op must not run for non-array imgs"),
    )
    labels = {"img": None}
    out = stage1._RandomGrayscaleTransform(p=1.0)(labels)
    assert out is labels and out["img"] is None


# ---------------------------------------------------------------------------
# (e) Golden polish dataset builder
# ---------------------------------------------------------------------------

def _make_sliced_tree(root: Path) -> Path:
    """Synthetic stage1_sliced-style tree: 3 izutsumi + 2 other train slices."""
    dataset_dir = root / "stage1_sliced"
    for split in ("train", "val", "test"):
        (dataset_dir / "images" / split).mkdir(parents=True)
        (dataset_dir / "labels" / split).mkdir(parents=True)

    def add(split: str, stem: str) -> None:
        (dataset_dir / "images" / split / f"{stem}.jpg").write_bytes(
            b"\xff\xd8\xff" + stem.encode()
        )
        (dataset_dir / "labels" / split / f"{stem}.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n"
        )

    for i in range(3):
        add("train", f"izutsumi_p01_s{i}")
    add("train", "manga109_p05_s0")
    add("train", "ahv1_s2")
    add("val", "izutsumi_p07_s0")
    add("val", "manga109_p02_s1")
    add("test", "manga109_p03_s0")
    add("test", "izutsumi_p09_s0")

    # The exact format src/training/slicing.py:write_sliced_dataset_yaml emits.
    (dataset_dir / "dataset.yaml").write_text(
        "# NOTE: ultralytics resolves relative paths from CWD, not from this file's location.\n"
        f"path: {dataset_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\nnames:\n"
        "  0: body\n"
        "  1: head\n"
        "  2: face\n",
        encoding="utf-8",
    )
    return dataset_dir


def test_build_polish_dataset_oversamples_train_izutsumi_x3(tmp_path: Path) -> None:
    dataset_dir = _make_sliced_tree(tmp_path)
    out_dir = tmp_path / "stage1_sliced_polish"

    stats = stage1._build_polish_dataset(dataset_dir, out_dir)

    assert stats["train_originals"] == 5
    assert stats["izutsumi_train"] == 3
    assert stats["izutsumi_train_effective"] == 9  # factor 3
    assert stats["duplicates_added"] == 6          # (factor - 1) x izutsumi
    assert stats["train_images"] == 11             # 5 originals + 6 dups
    assert stats["val_images"] == 2
    assert stats["test_images"] == 2

    train_files = sorted(p.name for p in (out_dir / "images" / "train").iterdir())
    assert len(train_files) == 11
    assert "izutsumi_p01_s0_polishdup1.jpg" in train_files
    assert "izutsumi_p01_s0_polishdup2.jpg" in train_files

    # Duplicate labels are real files carrying the source label content.
    dup_label = out_dir / "labels" / "train" / "izutsumi_p01_s0_polishdup1.txt"
    assert dup_label.is_file() and not dup_label.is_symlink()
    assert (dup_label.read_text()
            == (dataset_dir / "labels" / "train" / "izutsumi_p01_s0.txt").read_text())

    # Originals are relative symlinks resolving to the source slices.
    original = out_dir / "images" / "train" / "manga109_p05_s0.jpg"
    assert original.is_symlink()
    assert original.resolve() == (dataset_dir / "images" / "train" / "manga109_p05_s0.jpg").resolve()
    # Non-izutsumi train slices stay single.
    assert [p.name for p in (out_dir / "images" / "train").iterdir()
            if p.name.startswith("manga109")] == ["manga109_p05_s0.jpg"]


def test_build_polish_dataset_val_and_test_untouched(tmp_path: Path) -> None:
    dataset_dir = _make_sliced_tree(tmp_path)
    out_dir = tmp_path / "p"
    stage1._build_polish_dataset(dataset_dir, out_dir)

    for split in ("val", "test"):
        split_files = sorted((out_dir / "images" / split).iterdir())
        assert len(split_files) == (2 if split == "val" else 2)
        assert all("_polishdup" not in p.name for p in split_files)
        # izutsumi slices in val/test are mirrored, never duplicated.
    val_names = [p.name for p in (out_dir / "images" / "val").iterdir()]
    assert sorted(val_names) == ["izutsumi_p07_s0.jpg", "manga109_p02_s1.jpg"]


def test_build_polish_dataset_rewrites_yaml_path_only(tmp_path: Path) -> None:
    dataset_dir = _make_sliced_tree(tmp_path)
    out_dir = tmp_path / "p"
    stage1._build_polish_dataset(dataset_dir, out_dir)

    yaml_lines = (out_dir / "dataset.yaml").read_text(encoding="utf-8").splitlines()
    assert f"path: {out_dir.resolve()}" in yaml_lines
    assert "NOTE" in yaml_lines[0]  # header comment preserved byte-cmp below
    assert "train: images/train" in yaml_lines
    assert "val: images/val" in yaml_lines
    assert "names:" in yaml_lines
    source_lines = (dataset_dir / "dataset.yaml").read_text(encoding="utf-8").splitlines()
    for source_line, out_line in zip(source_lines, yaml_lines):
        if source_line.startswith("path:"):
            assert out_line != source_line  # only the path line changes
        else:
            assert out_line == source_line


def test_build_polish_dataset_refuses_rebuild_without_flag(tmp_path: Path) -> None:
    dataset_dir = _make_sliced_tree(tmp_path)
    out_dir = tmp_path / "p"
    stage1._build_polish_dataset(dataset_dir, out_dir)

    with pytest.raises(FileExistsError, match="polish-rebuild"):
        stage1._build_polish_dataset(dataset_dir, out_dir)

    stats = stage1._build_polish_dataset(dataset_dir, out_dir, rebuild=True)
    assert stats["duplicates_added"] == 6
    assert stats["train_images"] == 11  # rebuilt, not stacked onto old dups


def test_build_polish_dataset_skips_stale_label_caches(tmp_path: Path) -> None:
    dataset_dir = _make_sliced_tree(tmp_path)
    # ultralytics .cache scans embed absolute paths of the SOURCE tree: they
    # must not be mirrored into the polish tree or reused there.
    (dataset_dir / "labels" / "train" / "labels.cache").write_text("{}")

    stats = stage1._build_polish_dataset(dataset_dir, tmp_path / "p")

    assert stats["train_images"] == 11
    assert not (tmp_path / "p" / "labels" / "train" / "labels.cache").exists()


def test_build_polish_dataset_factor_two(tmp_path: Path) -> None:
    dataset_dir = _make_sliced_tree(tmp_path)
    stats = stage1._build_polish_dataset(dataset_dir, tmp_path / "p", factor=2)
    assert stats["duplicates_added"] == 3
    assert stats["izutsumi_train_effective"] == 6


def test_build_polish_dataset_rejects_bad_factor(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="factor"):
        stage1._build_polish_dataset(tmp_path, tmp_path / "p", factor=0)


def test_build_polish_dataset_missing_izutsumi_label_fails_loud(tmp_path: Path) -> None:
    dataset_dir = _make_sliced_tree(tmp_path)
    (dataset_dir / "labels" / "train" / "izutsumi_p01_s0.txt").unlink()
    with pytest.raises(FileNotFoundError, match="izutsumi"):
        stage1._build_polish_dataset(dataset_dir, tmp_path / "p")


def test_rewrite_dataset_yaml_path_roundtrip_format(tmp_path: Path) -> None:
    text = (
        "# NOTE: ultralytics resolves relative paths from CWD.\n"
        f"path: {tmp_path}/old\n"
        "train: images/train\n"
    )
    out = stage1._rewrite_dataset_yaml_path(text, str(tmp_path / "new"))
    assert f"path: {tmp_path}/new" in out
    assert "train: images/train" in out
    assert out.count("\n") == text.count("\n")
    with pytest.raises(ValueError, match="path"):
        stage1._rewrite_dataset_yaml_path("train: images/train\n", str(tmp_path))


# ---------------------------------------------------------------------------
# Polished hyperparameter profile
# ---------------------------------------------------------------------------

def test_polish_profile_forces_defaults_when_not_overridden() -> None:
    polished = stage1._apply_polish_profile(_stage1_params(), overrides=[])
    assert polished.epochs == 10
    assert polished.lr0 == pytest.approx(1e-4)
    assert polished.mosaic == 0.0
    assert polished.cos_lr is False
    assert polished.close_mosaic == 0
    # Non-forced fields pass through untouched.
    assert polished.batch == get_settings().training.stage1.batch


def test_polish_profile_respects_explicit_overrides() -> None:
    # _apply_overrides lands CLI overrides on the settings first; the profile
    # then must not re-force exactly those keys (main() applies it in order).
    overridden_params = _stage1_params(epochs=3)
    polished = stage1._apply_polish_profile(
        overridden_params, overrides=["training.stage1.epochs=3"]
    )
    assert polished.epochs == 3
    assert polished.lr0 == pytest.approx(1e-4)  # others still forced


# ---------------------------------------------------------------------------
# (f) CLI guards: --polish
# ---------------------------------------------------------------------------

def test_polish_without_resume_exits_2_before_any_load(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(sys, "argv", ["stage1.py", "--polish"])

    def fail_if_reached(*args, **kwargs):
        raise AssertionError("reached a model/config load; guard must exit first")

    monkeypatch.setattr(stage1, "YOLO", fail_if_reached)
    monkeypatch.setattr(stage1, "_apply_overrides", fail_if_reached)

    # caplog (not capsys): _setup_logging may reuse a StreamHandler bound in
    # an earlier capture context, bypassing the current sys.stderr object.
    with pytest.raises(SystemExit) as exc:
        with caplog.at_level(logging.ERROR):
            stage1.main()

    assert exc.value.code == 2
    assert "--resume" in caplog.text


def test_polish_with_subset_conflict_exits_2(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(sys, "argv", [
        "stage1.py", "--polish", "--resume", "phase1-best.pt",
        "--override", "training.stage1.use_subset=true",
    ])
    monkeypatch.setattr(stage1, "YOLO", lambda *a, **k: pytest.fail("model load"))

    with pytest.raises(SystemExit) as exc:
        with caplog.at_level(logging.ERROR):
            stage1.main()

    assert exc.value.code == 2
    assert "use_subset" in caplog.text


def test_pixi_m3_task_pins_complete_v1_profile() -> None:
    """The train-stage1-m3 task must pin the FULL v1 profile (review B1).

    m3_chain.sh hardcodes EPOCHS=26 and refuses a checkpoint saved with a
    different epochs arg, so an unpinned epochs (config now says 75)
    breaks the chain after cycle 1; the 16 GB Air also cannot absorb the
    v2 augmentation or 8 workers. Assert every v1-critical knob so config
    drift cannot silently break the M3 chain again.
    """
    import re
    import tomllib

    repo_root = Path(__file__).resolve().parents[1]
    pixi = tomllib.loads((repo_root / "pixi.toml").read_text(encoding="utf-8"))
    task = pixi["tasks"]["train-stage1-m3"]

    pins = {}
    for part in task.split("--override ")[1:]:
        key, _, value = part.strip().partition("=")
        pins[key] = value

    expected_v1 = {
        "training.stage1.device": "mps",
        "training.stage1.use_subset": "true",
        "training.stage1.batch": "8",
        "training.stage1.workers": "4",
        "training.stage1.epochs": "26",
        "training.stage1.patience": "12",
        "training.stage1.close_mosaic": "6",
        "training.stage1.cls_pw": "0.6",
        "training.stage1.hsv_h": "0",
        "training.stage1.hsv_s": "0",
        "training.stage1.hsv_v": "0.3",
        "training.stage1.mixup": "0",
        "training.stage1.random_grayscale": "0",
    }
    for key, value in expected_v1.items():
        assert pins.get(key) == value, (
            f"train-stage1-m3 must pin {key}={value} (unpinned = v2 config "
            f"value would reach the 16 GB MPS chain); got {pins.get(key)!r}"
        )
    assert pins["training.stage1.model"].endswith("yolo26n.pt")
    # Empty naming pins restore the historical stage1_m3 names.
    assert "training.stage1.run_name" in pins
    assert "training.stage1.target_model_name" in pins

    # The epochs pin must stay in lockstep with the chain script's EPOCHS.
    chain = (repo_root / "scripts" / "train" / "m3_chain.sh").read_text(encoding="utf-8")
    match = re.search(r'EPOCHS="\$\{EPOCHS:-(\d+)\}"', chain)
    assert match, "m3_chain.sh EPOCHS default not found"
    assert pins["training.stage1.epochs"] == match.group(1), (
        "train-stage1-m3 epochs pin diverged from m3_chain.sh EPOCHS default"
    )


# ---------------------------------------------------------------------------
# (g) Dataset resolution & --dataset CLI argument
# ---------------------------------------------------------------------------

class _InterceptTrain(Exception):
    def __init__(self, data_arg: str) -> None:
        self.data_arg = data_arg


class _MockYOLO:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def add_callback(self, *args, **kwargs) -> None:
        pass

    def train(self, *args, **kwargs) -> None:
        raise _InterceptTrain(kwargs.get("data", ""))


def test_explicit_dataset_arg_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ds_dir = tmp_path / "custom_ds"
    ds_dir.mkdir()
    yaml_file = ds_dir / "dataset.yaml"
    yaml_file.write_text("path: custom\nnames: {0: body}\n")

    monkeypatch.setattr(stage1, "YOLO", _MockYOLO)
    monkeypatch.setattr(sys, "argv", [
        "stage1.py",
        "--dataset", str(ds_dir),
        "--override", "training.stage1.epochs=1",
        "--override", "training.stage1.use_subset=false",
    ])

    with pytest.raises(_InterceptTrain) as exc:
        stage1.main()
    assert Path(exc.value.data_arg).resolve() == yaml_file.resolve()


def test_explicit_dataset_arg_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ds_dir = tmp_path / "custom_ds"
    ds_dir.mkdir()
    yaml_file = ds_dir / "custom_dataset.yaml"
    yaml_file.write_text("path: custom\nnames: {0: body}\n")

    monkeypatch.setattr(stage1, "YOLO", _MockYOLO)
    monkeypatch.setattr(sys, "argv", [
        "stage1.py",
        "--dataset", str(yaml_file),
        "--override", "training.stage1.epochs=1",
        "--override", "training.stage1.use_subset=false",
    ])

    with pytest.raises(_InterceptTrain) as exc:
        stage1.main()
    assert Path(exc.value.data_arg).resolve() == yaml_file.resolve()


def test_explicit_dataset_arg_missing_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    nonexistent = tmp_path / "ghost_ds"
    monkeypatch.setattr(stage1, "YOLO", _MockYOLO)
    monkeypatch.setattr(sys, "argv", [
        "stage1.py",
        "--dataset", str(nonexistent),
        "--override", "training.stage1.use_subset=false",
    ])

    with pytest.raises(SystemExit) as exc:
        with caplog.at_level(logging.ERROR):
            stage1.main()
    assert exc.value.code == 1
    assert "not found" in caplog.text


def test_autodiscover_stage1_v2_sliced_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Setup a data root with stage1_v2_sliced but NO stage1_sliced
    training_dir = tmp_path / "training"
    v2_dir = training_dir / "stage1_v2_sliced"
    v2_dir.mkdir(parents=True)
    v2_yaml = v2_dir / "dataset.yaml"
    v2_yaml.write_text("path: v2\nnames: {0: body}\n")

    monkeypatch.setattr(stage1, "YOLO", _MockYOLO)
    monkeypatch.setattr(sys, "argv", [
        "stage1.py",
        "--override", f"paths.data={tmp_path}",
        "--override", f"paths.unified_dir={training_dir / 'stage1'}",
        "--override", f"paths.stage1_sliced_dir={training_dir / 'stage1_sliced'}",
        "--override", "training.stage1.epochs=1",
        "--override", "training.stage1.use_subset=false",
    ])

    with pytest.raises(_InterceptTrain) as exc:
        stage1.main()
    assert Path(exc.value.data_arg).resolve() == v2_yaml.resolve()


def test_patch_musgd_cublas_execution() -> None:
    import torch
    import ultralytics.optim.muon as muon_mod
    stage1._patch_musgd_cublas()
    # Test 2D, transposed 2D, and 3D
    for shape in [(32, 16), (16, 32), (4, 16, 16)]:
        t = torch.randn(*shape)
        out = muon_mod.zeropower_via_newtonschulz5(t)
        assert out.shape == t.shape
        assert out.dtype == t.dtype


# ---------------------------------------------------------------------------
# (g) save_best_model fallback for resumed ephemeral runs
# ---------------------------------------------------------------------------

def test_save_best_model_prefers_best_then_falls_back_to_last(tmp_path: Path) -> None:
    from src.training.preparation import save_best_model

    project_dir = tmp_path / "detect"
    weights_dir = project_dir / "stage1_v2" / "weights"
    weights_dir.mkdir(parents=True)
    target_dir = tmp_path / "models"

    # Both absent -> None
    assert save_best_model(project_dir, "stage1_v2", target_dir, "out.pt") is None

    # Only last.pt present (resumed VM didn't surpass prior best_fitness) -> falls back
    (weights_dir / "last.pt").write_bytes(b"last-weights")
    saved = save_best_model(project_dir, "stage1_v2", target_dir, "out.pt")
    assert saved == target_dir / "out.pt"
    assert saved.read_bytes() == b"last-weights"

    # best.pt present -> prefers best.pt over last.pt
    (weights_dir / "best.pt").write_bytes(b"best-weights")
    saved = save_best_model(project_dir, "stage1_v2", target_dir, "out.pt")
    assert saved.read_bytes() == b"best-weights"


