"""Unit tests for the Stage 2 SupCon fine-tuning script (scripts/train/stage2.py).

All tests are synthetic and CPU-only; nothing downloads weights and no real
training executes — helpers are exercised on tiny data, ``train`` only through
its guard clauses, and the two CLI paths behave as the contract pins.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from scripts.train.stage2 import (
    IZUTSUMI_LABEL,
    _resolve_device,
    embed_dataset,
    eval_separability,
    mine_pseudo_labels,
)
from src.reid.model import CharacterEmbedder

pytest.importorskip("cv2")


@pytest.fixture(scope="module")
def embedder() -> CharacterEmbedder:
    model = CharacterEmbedder(
        backbone="convnext_small", embedding_dim=256, gem_p=3.0, pretrained=False
    )
    return model.eval()


def _write_crop(path, color=(120, 140, 90)) -> None:
    rgb = np.zeros((12, 12, 3), dtype=np.uint8)
    rgb[:, :] = color
    Image.fromarray(rgb).save(path)


# ---------------------------------------------------------------------------
# (h) embed_dataset
# ---------------------------------------------------------------------------

def test_embed_dataset_three_synthetic_crops(tmp_path, embedder) -> None:
    paths = []
    for index in range(3):
        path = tmp_path / f"crop_{index}.png"
        _write_crop(path, color=(30 * index, 100, 220 - 30 * index))
        paths.append(path)

    embeddings = embed_dataset(embedder, paths)

    assert embeddings.shape == (3, embedder.embedding_dim)
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)


def test_embed_dataset_empty_list_returns_empty_matrix(embedder) -> None:
    embeddings = embed_dataset(embedder, [])

    assert embeddings.shape == (0, embedder.embedding_dim)


def test_embed_dataset_unreadable_crop_fails_loud(tmp_path, embedder) -> None:
    not_an_image = tmp_path / "corrupt.png"
    not_an_image.write_bytes(b"definitely not a png")

    with pytest.raises(ValueError, match="Could not read image"):
        embed_dataset(embedder, [not_an_image])


# ---------------------------------------------------------------------------
# (g) mine_pseudo_labels
# ---------------------------------------------------------------------------

def _seeded_unit_blobs() -> np.ndarray:
    """Two well-separated gaussian direction blobs + 4 far scattered singletons."""
    def _blob(direction: np.ndarray, spread: float, count: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        points = np.repeat(direction[None, :], count, axis=0)
        points = points + rng.normal(0.0, spread, (count, 2))
        return points / np.linalg.norm(points, axis=1, keepdims=True)

    blob_a = _blob(np.array([1.0, 0.0]), spread=0.03, count=12, seed=10)
    blob_b = _blob(np.array([0.0, 1.0]), spread=0.03, count=12, seed=11)
    singletons = np.array(
        [[-1.0, 0.0], [0.0, -1.0], [-0.8, -0.6], [-0.6, -0.8]], dtype=np.float64
    )
    singletons = singletons / np.linalg.norm(singletons, axis=1, keepdims=True)
    return np.vstack([blob_a, blob_b, singletons]).astype(np.float32)


def test_mine_pseudo_labels_two_blobs_and_unique_singletons() -> None:
    embeddings = _seeded_unit_blobs()

    labels = mine_pseudo_labels(embeddings, min_cluster_size=5)

    assert len(labels) == 28
    assert 0 not in labels  # 0 is reserved for izutsumi

    blob_a, blob_b = labels[:12], labels[12:24]
    singles = labels[24:]
    assert len(set(blob_a)) == 1       # blob internal cohesion
    assert len(set(blob_b)) == 1
    assert set(blob_a) != set(blob_b)  # blobs split from each other
    assert len(set(singles)) == 4      # every singleton unique
    assert set(singles).isdisjoint(set(blob_a) | set(blob_b))


def test_mine_pseudo_labels_short_input_all_singletons() -> None:
    labels = mine_pseudo_labels(np.eye(4, dtype=np.float32), min_cluster_size=5)
    assert labels == [1, 2, 3, 4]


def test_mine_pseudo_labels_empty_input() -> None:
    assert mine_pseudo_labels(np.zeros((0, 8), dtype=np.float32)) == []


def test_mine_pseudo_labels_rejects_wrong_rank() -> None:
    with pytest.raises(ValueError, match="must be"):
        mine_pseudo_labels(np.zeros((3, 4, 5), dtype=np.float32))


# ---------------------------------------------------------------------------
# eval_separability
# ---------------------------------------------------------------------------

def test_eval_separability_hand_computed() -> None:
    embeddings = np.array(
        [[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    # izu-izu mean = 0.6, izu-other mean = 0.0 -> separability 0.6
    separability = eval_separability(embeddings, [True, True, False])

    assert separability == pytest.approx(0.6, abs=1e-6)


def test_eval_separability_requires_two_izutsumi_and_one_other() -> None:
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="separability needs"):
        eval_separability(embeddings, [True, False])


# ---------------------------------------------------------------------------
# train() guard clauses (no real training in the test suite)
# ---------------------------------------------------------------------------

def _train_guard_kwargs(model_path, num_classes=3, **overrides):
    return {
        "num_classes": num_classes,
        "device": torch.device("cpu"),
        "batch_size": 4,
        "epochs": 1,
        "learning_rate": 1e-3,
        "patience": 1,
        "temperature": 0.07,
        "num_subcenters": 5,
        "model_path": model_path,
        **overrides,
    }


def test_train_rejects_empty_inputs(tmp_path) -> None:
    from scripts.train.stage2 import train

    with pytest.raises(ValueError, match="non-empty and aligned"):
        train(
            CharacterEmbedder(pretrained=False),
            [],
            [],
            None,
            None,
            **_train_guard_kwargs(tmp_path / "m.pt"),
        )


def test_train_rejects_labels_paths_misalignment(tmp_path) -> None:
    from scripts.train.stage2 import train

    with pytest.raises(ValueError, match="non-empty and aligned"):
        train(
            CharacterEmbedder(pretrained=False),
            [tmp_path / "a.png", tmp_path / "b.png"],
            [0],
            None,
            None,
            **_train_guard_kwargs(tmp_path / "m.pt"),
        )


def test_train_rejects_unusable_validation_split(tmp_path) -> None:
    from scripts.train.stage2 import train

    with pytest.raises(ValueError, match="validation split unusable"):
        train(
            CharacterEmbedder(pretrained=False),
            [tmp_path / "a.png"],
            [0],
            [tmp_path / "v.png"],
            [IZUTSUMI_LABEL],  # only 1 izutsumi val crop (<2)
            **_train_guard_kwargs(tmp_path / "m.pt"),
        )


def test_train_requires_two_classes() -> None:
    from scripts.train.stage2 import train

    with pytest.raises(ValueError, match="num_classes must be >= 2"):
        train(
            CharacterEmbedder(pretrained=False),
            [None],  # never reached — guard fires first
            [0],
            None,
            None,
            **_train_guard_kwargs("unused.pt", num_classes=1),
        )


def test_resolve_device_falls_back_with_warning() -> None:
    device = _resolve_device("cuda")  # never available on CPU test machines
    assert device.type == "cpu"


# ---------------------------------------------------------------------------
# (i) CLI paths
# ---------------------------------------------------------------------------

def test_main_missing_stage2_dir_exits_one_with_actionable_message(
    tmp_path, caplog
) -> None:
    import logging

    from scripts.train import stage2

    caplog.set_level(logging.INFO)
    missing = tmp_path / "missing_stage2"
    argv = ["--override", f"paths.stage2_dir={missing}"]

    with pytest.raises(SystemExit) as excinfo:
        stage2.main(argv)

    assert excinfo.value.code == 1
    assert "does not exist" in caplog.text
    assert "--prepare-only" in caplog.text  # actionable next step


def test_main_prepare_only_exits_zero_and_monkeypatches_prep(
    monkeypatch, tmp_path
) -> None:
    from scripts.train import stage2

    called = {}

    def _fake_prepare_triplet_dataset(source_dirs, **kwargs):
        called["source_dirs"] = list(source_dirs)
        return {
            "train": {"izutsumi": 3, "not_izutsumi": 5},
            "val": {"izutsumi": 1, "not_izutsumi": 2},
        }

    monkeypatch.setattr(stage2, "prepare_triplet_dataset", _fake_prepare_triplet_dataset)

    crops_dir = tmp_path / "crops"
    crops_dir.mkdir()

    with pytest.raises(SystemExit) as excinfo:
        stage2.main(
            ["--prepare-only", "--override", f"paths.crops_dir={crops_dir}"]
        )

    assert excinfo.value.code == 0
    assert called["source_dirs"] == [crops_dir]


def test_mine_pseudo_labels_fails_loud_above_dense_budget(monkeypatch) -> None:
    """Dense precomputed-distance mining is loudly capped (review M5).

    Two N x N float32 arrays at ~50k not_izutsumi crops would OOM a long
    Colab run; the guard mirrors the budget already enforced by
    src.reid.clustering.subcluster_embeddings.
    """
    import numpy as np

    from src.reid import clustering

    monkeypatch.setattr(clustering, "MAX_SUBCLUSTER_INSTANCES", 4)
    embeddings = np.tile(np.linspace(0.0, 1.0, 16).reshape(-1, 1), (1, 4))

    with pytest.raises(ValueError, match="dense precomputed HDBSCAN budget"):
        mine_pseudo_labels(embeddings, min_cluster_size=2)
