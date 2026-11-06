"""Unit tests for SupConLoss and SubCenterArcFace (src/reid/loss.py).

All tests are synthetic and CPU-only.
"""

import math

import pytest
import torch

from src.reid.loss import SubCenterArcFace, SupConLoss


def _normalized_features(batch: int, dim: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    features = torch.randn(batch, dim)
    return torch.nn.functional.normalize(features, dim=1)


# ---------------------------------------------------------------------------
# (e) SupConLoss
# ---------------------------------------------------------------------------

def test_supcon_identical_same_label_pair_loss_is_zero() -> None:
    loss_fn = SupConLoss(temperature=0.07)
    features = torch.nn.functional.normalize(torch.ones(2, 8), dim=1)
    labels = torch.tensor([0, 0])

    loss = loss_fn(features, labels)

    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_supcon_all_unique_labels_loss_is_exactly_zero() -> None:
    loss_fn = SupConLoss(temperature=0.07)
    features = _normalized_features(batch=6, dim=8, seed=1)
    labels = torch.arange(6)

    loss = loss_fn(features, labels)

    assert float(loss) == 0.0
    assert not torch.isnan(loss)


def test_supcon_labels_none_acts_as_instance_discrimination() -> None:
    loss_fn = SupConLoss(temperature=0.07)
    features = _normalized_features(batch=6, dim=8, seed=2)

    with_labels = loss_fn(features, None)
    expected = loss_fn(features, torch.arange(6))

    assert float(with_labels) == pytest.approx(float(expected), rel=1e-6)


def test_supcon_multi_member_cluster_loss_matches_partition_formula() -> None:
    # Every anchor: positives identical to each other (cos 1), a far negative
    # class at cos 0 -> per-anchor -log(exp(1/tau)/((B-1)exp(1/tau) + neg)).
    # With negatives at cos 0: loss = log((B-1) + exp(-1/tau)).
    tau = 0.07
    loss_fn = SupConLoss(temperature=tau)
    features = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    labels = torch.tensor([0, 0, 0, 1])

    loss = loss_fn(features, labels)

    # Anchors 0..2 have positives; anchor 3 has none (dropped).
    expected = math.log(3 - 1 + math.exp(-1.0 / tau))
    assert float(loss) == pytest.approx(expected, rel=1e-3)


def test_supcon_gradient_flows_and_is_finite() -> None:
    loss_fn = SupConLoss(temperature=0.07)
    features = _normalized_features(batch=5, dim=4, seed=3).requires_grad_(True)
    labels = torch.tensor([0, 0, 1, 1, 2])

    loss = loss_fn(features, labels)
    loss.backward()

    assert torch.isfinite(loss).all()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert float(features.grad.abs().sum()) > 0.0


def test_supcon_invalid_temperature_raises() -> None:
    with pytest.raises(ValueError, match="temperature"):
        SupConLoss(temperature=0.0)


def test_supcon_two_dim_features_enforced() -> None:
    loss_fn = SupConLoss()
    with pytest.raises(ValueError, match="features must be"):
        loss_fn(torch.randn(2, 2, 2), torch.zeros(2, dtype=torch.long))


# ---------------------------------------------------------------------------
# (f) SubCenterArcFace
# ---------------------------------------------------------------------------

def _arcface_pair(num_classes: int = 4, in_features: int = 32, k: int = 3):
    torch.manual_seed(4)
    margin_head = SubCenterArcFace(in_features, num_classes, num_subcenters=k, m=0.5)
    torch.manual_seed(4)
    plain_head = SubCenterArcFace(in_features, num_classes, num_subcenters=k, m=0.0)
    # Identical sub-center weights: only the margin differs.
    plain_head.load_state_dict(margin_head.state_dict())
    return margin_head, plain_head


def test_arcface_logits_shape_and_finite_loss() -> None:
    margin_head, _ = _arcface_pair()
    features = _normalized_features(batch=6, dim=32, seed=5)
    labels = torch.tensor([0, 1, 2, 3, 0, 1])

    logits, loss = margin_head(features, labels)

    assert logits.shape == (6, 4)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
    assert float(loss) > 0.0


def test_arcface_margin_increases_loss_over_m_zero() -> None:
    margin_head, plain_head = _arcface_pair()
    features = _normalized_features(batch=8, dim=32, seed=6)
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])

    _, loss_margin = margin_head(features, labels)
    _, loss_plain = plain_head(features, labels)

    assert float(loss_margin) > float(loss_plain)


def test_arcface_subcenter_max_selection_on_non_target_classes() -> None:
    # With m=0, logits = s * max_k cosine; verify against hand-computed max
    # over sub-centers for classes that receive no margin.
    plain_head = SubCenterArcFace(16, 4, num_subcenters=3, m=0.0)

    with torch.no_grad():
        features = _normalized_features(batch=5, dim=16, seed=7)
        labels = torch.tensor([0, 0, 0, 0, 0])
        logits, _ = plain_head(features, labels)

        weight = torch.nn.functional.normalize(plain_head.weight, dim=1)
        cosine = (features @ weight.T).view(5, 4, 3)  # [B, C, K]
        for row in range(5):
            for cls in (1, 2, 3):  # non-target: pure subcenter max * scale
                expected = plain_head.s * cosine[row, cls].max()
                assert float(logits[row, cls]) == pytest.approx(float(expected), rel=1e-5)


def test_arcface_out_of_range_labels_raise() -> None:
    margin_head, _ = _arcface_pair(num_classes=3)
    features = _normalized_features(batch=2, dim=32, seed=8)
    labels = torch.tensor([0, 3])

    with pytest.raises(ValueError, match="labels must lie in"):
        margin_head(features, labels)


def test_arcface_margin_must_be_valid() -> None:
    with pytest.raises(ValueError, match="margin m"):
        SubCenterArcFace(8, 2, m=math.pi)
