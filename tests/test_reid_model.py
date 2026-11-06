"""Unit tests for the Stage 2 character embedder (src/reid/model.py).

All tests are synthetic, CPU-only, and use pretrained=False so nothing ever
touches the network.
"""

import numpy as np
import pytest
import torch

from src.reid.model import (
    BACKBONES,
    CharacterEmbedder,
    GeneralizedMeanPooling,
    fuse_part_embeddings,
    load_embedder,
    save_embedder,
)

PART_WEIGHTS = {"face": 0.35, "head": 0.50, "body": 0.15}


def _random_unit_vector(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dim).astype(np.float32)
    return vector / np.linalg.norm(vector)


def _build_embedder(embedding_dim: int = 256, gem_p: float = 3.0) -> CharacterEmbedder:
    return CharacterEmbedder(
        backbone="convnext_small",
        embedding_dim=embedding_dim,
        gem_p=gem_p,
        pretrained=False,
    )


# ---------------------------------------------------------------------------
# (a) CharacterEmbedder basics
# ---------------------------------------------------------------------------

def test_forward_shape_and_l2_normalization() -> None:
    torch.manual_seed(0)
    embedder = _build_embedder().eval()
    batch = torch.randn(4, 3, 224, 224)

    with torch.no_grad():
        embeddings = embedder(batch)

    assert embeddings.shape == (4, 256)
    norms = embeddings.norm(dim=1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)


def test_embed_single_returns_unit_vector() -> None:
    embedder = _build_embedder().eval()
    image = np.random.default_rng(1).integers(0, 255, size=(96, 72, 3), dtype=np.uint8)

    embedding = embedder.embed_single(image)

    assert embedding.shape == (256,)
    assert embedding.dtype == np.float32
    assert float(np.linalg.norm(embedding)) == pytest.approx(1.0, abs=1e-5)


def test_gem_p_is_learnable_and_initialized_from_config() -> None:
    embedder = _build_embedder(gem_p=3.0)
    pool_parameters = [name for name, _ in embedder.pool.named_parameters()]

    assert pool_parameters == ["p"]
    assert embedder.pool.p.requires_grad is True
    assert float(embedder.pool.p) == pytest.approx(3.0)


def test_gem_forward_clamps_p_above_one() -> None:
    torch.manual_seed(2)
    feature_map = torch.randn(2, 4, 5, 5)

    clamped_pool = GeneralizedMeanPooling(p=0.1)
    clamped_pool.p.data.fill_(0.1)  # simulate a drifted parameter
    exact_p1_pool = GeneralizedMeanPooling(p=1.0)

    with torch.no_grad():
        pooled_clamped = clamped_pool(feature_map)
        pooled_p1 = exact_p1_pool(feature_map)

    assert torch.isfinite(pooled_clamped).all()
    assert torch.allclose(pooled_clamped, pooled_p1, atol=1e-6)


def test_constructor_signature_matches_contract() -> None:
    import inspect

    parameters = inspect.signature(CharacterEmbedder.__init__).parameters
    assert list(parameters) == ["self", "backbone", "embedding_dim", "gem_p", "pretrained"]
    assert parameters["backbone"].default == "convnext_small"
    assert parameters["embedding_dim"].default == 256
    assert parameters["gem_p"].default == 3.0
    assert parameters["pretrained"].default is True


def test_unknown_backbone_fails_loud() -> None:
    with pytest.raises(ValueError, match="Unknown backbone"):
        CharacterEmbedder(backbone="resnet999", pretrained=False)


def test_backbone_registry_contents() -> None:
    assert set(BACKBONES) == {"convnext_small", "dinov2_vits14", "ccip"}


def test_dinov2_unavailable_hub_raises_construction_error(monkeypatch) -> None:
    def _boom(*args, **kwargs):
        raise OSError("no network")

    monkeypatch.setattr("torch.hub.load", _boom)
    with pytest.raises(RuntimeError, match="dinov2_vits14.*torch.hub"):
        CharacterEmbedder(backbone="dinov2_vits14", pretrained=False)


def test_ccip_backbone_never_fakes() -> None:
    with pytest.raises(NotImplementedError, match="ccip"):
        CharacterEmbedder(backbone="ccip", pretrained=False)


# ---------------------------------------------------------------------------
# (b) Multi-granularity fusion
# ---------------------------------------------------------------------------

def test_fusion_hand_computed_direction() -> None:
    e_face = _random_unit_vector(6, seed=3)
    e_head = _random_unit_vector(6, seed=4)

    fused = fuse_part_embeddings(
        parts={"face": e_face, "head": e_head},
        scores={"face": 0.9, "head": 0.8},
        weights=PART_WEIGHTS,
    )

    weights_present = np.array([0.35, 0.5])
    renorm = weights_present / weights_present.sum()
    expected = (renorm * np.array([0.9, 0.8]))[:, None] * np.stack([e_face, e_head])
    expected = expected.sum(axis=0)
    expected = expected / np.linalg.norm(expected)

    assert fused == pytest.approx(expected.astype(np.float32), rel=1e-5, abs=1e-6)
    assert float(np.linalg.norm(fused)) == pytest.approx(1.0, abs=1e-6)


def test_fusion_subset_renormalized_over_present_parts() -> None:
    e_head = _random_unit_vector(8, seed=5)

    fused = fuse_part_embeddings(
        parts={"head": e_head},
        scores={"head": 0.8},
        weights=PART_WEIGHTS,
    )

    # Weight renormalization + positive-confidence scaling never change the
    # direction: a head-only instance embeds as the head embedding itself.
    assert fused == pytest.approx(e_head, abs=1e-6)


def test_fusion_parts_scores_key_mismatch_raises() -> None:
    e_face = _random_unit_vector(6, seed=6)
    with pytest.raises(ValueError, match="keys must match exactly"):
        fuse_part_embeddings(
            parts={"face": e_face},
            scores={"head": 0.9},
            weights=PART_WEIGHTS,
        )


def test_fusion_missing_weight_for_present_part_raises() -> None:
    e_face = _random_unit_vector(6, seed=7)
    e_head = _random_unit_vector(6, seed=8)
    with pytest.raises(ValueError, match="No fusion weight configured"):
        fuse_part_embeddings(
            parts={"face": e_face, "head": e_head},
            scores={"face": 0.9, "head": 0.8},
            weights={"face": 0.35},
        )


def test_fusion_empty_parts_raises() -> None:
    with pytest.raises(ValueError, match="at least one part"):
        fuse_part_embeddings(parts={}, scores={}, weights=PART_WEIGHTS)


def test_fusion_zero_confidence_everywhere_raises() -> None:
    e_face = _random_unit_vector(6, seed=9)
    with pytest.raises(ValueError, match="zero magnitude"):
        fuse_part_embeddings(
            parts={"face": e_face}, scores={"face": 0.0}, weights=PART_WEIGHTS
        )


# ---------------------------------------------------------------------------
# (c) Save / load roundtrip
# ---------------------------------------------------------------------------

def test_save_and_load_embedder_roundtrip(tmp_path) -> None:
    torch.manual_seed(10)
    embedder = _build_embedder().eval()
    image = np.random.default_rng(11).integers(0, 255, size=(64, 64, 3), dtype=np.uint8)

    before = embedder.embed_single(image)
    model_path = tmp_path / "nested" / "reid_stage2_v2.pt"
    save_embedder(embedder, model_path)

    assert model_path.exists()
    restored = load_embedder(model_path)
    assert restored.training is False  # load returns an inference-ready model

    after = restored.embed_single(image)
    np.testing.assert_allclose(before, after, rtol=0, atol=0)


def test_load_embedder_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        load_embedder("/nonexistent/path/reid.pt")
