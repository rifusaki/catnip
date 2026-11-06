"""Unit tests for Stage 2 semi-supervised multi-prototype clustering.

All tests are synthetic-data only (no model files, no network, no dataset),
CPU-runnable, and exercise src/reid.clustering against exact brute-force
references.
"""

import logging
from pathlib import Path

import numpy as np
import pytest

from src.config import ClusteringParams
from src.reid import clustering
from src.reid.clustering import (
    PrototypeBank,
    apply_page_priors,
    apply_panel_constraints,
    assign_izutsumi,
    assign_two_tier,
    build_review_gallery,
    knn_graph,
    l2_normalize,
    medoid_of,
    subcluster_embeddings,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_params(**overrides) -> ClusteringParams:
    """ClusteringParams with defaults, minus test-specific overrides."""
    defaults = dict(
        k_neighbors=25,
        min_cluster_size=5,
        tau_core=0.82,
        tau_boundary=0.68,
        page_prior_boost=0.06,
        panel_instance_iou=0.15,
    )
    defaults.update(overrides)
    return ClusteringParams(**defaults)


def unit(vector) -> np.ndarray:
    """L2-normalize a 1-D vector."""
    vector = np.asarray(vector, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def build_bank(positive_directions, negative_directions, k=5, mcs=5) -> PrototypeBank:
    bank = PrototypeBank(k_neighbors=k, min_cluster_size=mcs)
    for direction in positive_directions:
        bank.add_must_link(unit(direction))
    for direction in negative_directions:
        bank.add_cannot_link(unit(direction))
    bank.fit_subclusters()
    return bank


def seeded_blobs(seed: int = 13, dim: int = 16, n_per_blob: int = 40):
    """Two 60-degree-separated dense blobs + 4 orthogonal-unit outliers.

    The geometry matters: the blobs merge with each other well before the
    outliers attach to anything, which is exactly the regime where a lone
    point must be reported as HDBSCAN noise.
    """
    rng = np.random.default_rng(seed)
    d_a = np.zeros(dim); d_a[0] = np.cos(np.deg2rad(30)); d_a[1] = np.sin(np.deg2rad(30))
    d_b = np.zeros(dim); d_b[0] = np.cos(np.deg2rad(30)); d_b[1] = -np.sin(np.deg2rad(30))
    blob_a = d_a + rng.normal(0, 0.02, (n_per_blob, dim))
    blob_b = d_b + rng.normal(0, 0.02, (n_per_blob, dim))
    outliers = np.eye(dim)[2:6]
    X = np.vstack([blob_a, blob_b, outliers])
    return X / np.linalg.norm(X, axis=1, keepdims=True), 2 * n_per_blob


# ---------------------------------------------------------------------------
# knn_graph — exact numpy path vs brute force
# ---------------------------------------------------------------------------


def _brute_force_knn(matrix: np.ndarray, k: int):
    """Reference top-k: similarity descending, ties by ascending index."""
    sims = matrix @ matrix.T
    np.fill_diagonal(sims, -np.inf)
    n = matrix.shape[0]
    indices = np.empty((n, k), dtype=np.int64)
    similarities = np.empty((n, k), dtype=np.float64)
    for row in range(n):
        order = sorted(range(n), key=lambda j: (-sims[row, j], j))[:k]
        indices[row] = order
        similarities[row] = sims[row, order]
    return indices, similarities


def test_knn_graph_matches_brute_force_exact():
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(12, 8))
    matrix = clustering._as_embedding_matrix(matrix)
    indices = knn_graph(matrix, k=4, use_faiss_if_available=False)
    expected_idx, expected_sims = _brute_force_knn(matrix, 4)

    assert indices.shape == (12, 4)
    assert np.array_equal(indices, expected_idx)

    # faiss is not installed in this environment; the default call must fall
    # back to the same exact numpy path.
    assert np.array_equal(knn_graph(matrix, k=4), expected_idx)


def test_knn_graph_similarity_ordering_and_self_exclusion():
    rng = np.random.default_rng(7)
    matrix = clustering._as_embedding_matrix(rng.normal(size=(30, 16)))
    indices = knn_graph(matrix, k=6, use_faiss_if_available=False)
    sims = np.take_along_axis(matrix @ matrix.T, indices, axis=1)

    # Self is never returned; rows are similarity-descending (tie order by
    # ascending index is verified by the brute-force test above).
    assert not np.any(indices == np.arange(30)[:, None])
    assert np.all(np.diff(sims, axis=1) <= 1e-12)
    for row in range(30):
        assert len(set(indices[row].tolist())) == 6


def test_knn_graph_clips_k_below_n_minus_one():
    matrix = clustering._as_embedding_matrix(np.random.default_rng(1).normal(size=(12, 5)))
    indices = knn_graph(matrix, k=50, use_faiss_if_available=False)
    assert indices.shape == (12, 11)


def test_knn_graph_rejects_invalid_k():
    matrix = clustering._as_embedding_matrix(np.eye(4))
    with pytest.raises(ValueError, match="k must be >= 1"):
        knn_graph(matrix, k=0)


def test_l2_normalize_rejects_zero_rows_and_bad_shapes():
    with pytest.raises(ValueError, match="zero-norm"):
        l2_normalize(np.array([[1.0, 0.0], [0.0, 0.0]]))
    with pytest.raises(ValueError, match="shape"):
        l2_normalize(np.zeros((2, 3, 4)))
    # 1-D and 2-D both normalize unit-length.
    assert np.isclose(np.linalg.norm(l2_normalize(np.array([3.0, 4.0, 0.0]))), 1.0)
    assert np.allclose(np.linalg.norm(l2_normalize(np.eye(3)), axis=1), 1.0)


# ---------------------------------------------------------------------------
# subcluster_embeddings
# ---------------------------------------------------------------------------


def test_subclusters_two_blobs_with_outliers_noise():
    X, blob_n = seeded_blobs()
    labels = subcluster_embeddings(X, k_neighbors=10, min_cluster_size=5)

    blob_a, blob_b, outliers = labels[:blob_n // 2], labels[blob_n // 2:blob_n], labels[blob_n:]
    # exactly two sub-clusters
    assert sorted(set(labels.tolist()) - {-1}) == [0, 1]
    # each blob is one uniform, non-noise sub-cluster, and the two differ
    assert len(set(blob_a.tolist())) == 1 and blob_a[0] >= 0
    assert len(set(blob_b.tolist())) == 1 and blob_b[0] >= 0
    assert blob_a[0] != blob_b[0]
    # outliers are noise
    assert bool(np.all(outliers == -1))


def test_subclusters_sparse_knn_graph_bridges_components():
    """Sparse opt-in path on two fully disconnected blobs: >1 component in the
    raw k-NN graph must be bridged (not rejected) and still yield 2 clusters."""
    rng = np.random.default_rng(4)
    # Blobs live in orthogonal subspaces: every cross-blob distance is ~1.41.
    blob_a = np.hstack([np.zeros((30, 8)), rng.normal(0, 0.01, (30, 8))])
    blob_a[:, 8] = 1.0
    blob_b = np.hstack([rng.normal(0, 0.01, (30, 8)), np.zeros((30, 8))])
    blob_b[:, 0] = 1.0
    X = np.vstack([blob_a, blob_b])
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    labels = subcluster_embeddings(
        X, k_neighbors=3, min_cluster_size=2, use_sparse_knn_graph=True
    )
    assert sorted(set(labels.tolist()) - {-1}) == [0, 1]
    assert len(set(labels[:30].tolist())) == 1
    assert len(set(labels[30:].tolist())) == 1
    assert labels[0] != labels[30]


def test_subclusters_unimodal_set_is_one_cluster():
    rng = np.random.default_rng(9)
    seeds = np.tile([1.0, 0.0, 0.0], (8, 1)) + rng.normal(0, 0.02, (8, 3))
    seeds = seeds / np.linalg.norm(seeds, axis=1, keepdims=True)
    labels = subcluster_embeddings(seeds, k_neighbors=5, min_cluster_size=5)
    # allow_single_cluster: a single mode is one cluster, never uniformly noise.
    assert (labels == 0).sum() > 0


def test_subclusters_small_input_returns_all_noise(caplog):
    X = clustering._as_embedding_matrix(np.eye(4))
    with caplog.at_level(logging.WARNING, logger="catnip.reid.clustering"):
        labels = subcluster_embeddings(X, k_neighbors=3, min_cluster_size=5)
    assert labels.tolist() == [-1, -1, -1, -1]
    assert any("no density-supported cluster" in record.message for record in caplog.records)


def test_subclusters_rejects_oversized_input(monkeypatch):
    monkeypatch.setattr(clustering, "MAX_SUBCLUSTER_INSTANCES", 10)
    X = clustering._as_embedding_matrix(np.eye(12))
    with pytest.raises(ValueError, match="budget of 10"):
        subcluster_embeddings(X, k_neighbors=5, min_cluster_size=5)


# ---------------------------------------------------------------------------
# medoid_of
# ---------------------------------------------------------------------------


def test_medoid_of_matches_brute_force_and_handles_single():
    rng = np.random.default_rng(2)
    cluster = rng.normal(size=(9, 6))
    cluster = cluster / np.linalg.norm(cluster, axis=1, keepdims=True)

    sims = cluster @ cluster.T
    expected = int(np.argmax(sims.sum(axis=1)))
    assert medoid_of(cluster) == expected
    assert medoid_of(unit([1.0, 2.0, 3.0]).reshape(1, -1)) == 0


def test_medoid_identity_point_of_symmetric_cluster_wins():
    # A point sitting at the summed-similarity center of its cluster wins.
    a, b = unit([1.0, 1.0, 0.0]), unit([1.0, -1.0, 0.0])
    middle = unit(a + b)
    cluster = np.stack([a, b, middle])
    assert medoid_of(cluster) == 2


# ---------------------------------------------------------------------------
# PrototypeBank
# ---------------------------------------------------------------------------


def test_prototype_bank_multi_modal_prototypes_and_nearest():
    rng = np.random.default_rng(8)
    # Blobs merge with each other (60 deg) long before the far singleton
    # attaches, so the singleton must be reported as noise and become its
    # own prototype rather than being absorbed into a mode.
    d_a = np.array([1.0, 0.0, 0.0])
    d_b = np.array([0.5, np.sqrt(3) / 2, 0.0])
    singleton = -unit(d_a + d_b)
    seeds_a = d_a + rng.normal(0, 0.02, (6, 3))
    seeds_b = d_b + rng.normal(0, 0.02, (6, 3))
    bank = build_bank(
        positive_directions=[*seeds_a, *seeds_b, singleton],
        negative_directions=[],
        k=5, mcs=5,
    )

    prototypes = bank.prototypes
    # two density mode medoids + the verified singleton seed as its own prototype
    assert prototypes.shape == (3, 3)
    seeds = clustering._as_embedding_matrix(np.vstack([seeds_a, seeds_b, singleton]))
    for prototype in prototypes:
        assert any(np.allclose(prototype, seeds[row]) for row in range(len(seeds)))

    # nearest prototype: a query near blob B scores exactly like the brute
    # force max over prototypes (no negatives -> veto inactive).
    query = unit(d_b + np.array([0.0, 0.05, 0.0]))
    expected = float((prototypes @ query).max())
    assert bank.max_similarity(query) == pytest.approx(expected, abs=1e-12)


def test_prototype_bank_veto_rule_exact():
    """Documented rule: eff = s_pos if s_neg <= s_pos else 2*s_pos - s_neg."""
    pos = np.array([1.0, 0.0, 0.0])
    neg = np.array([0.0, 1.0, 0.0])
    bank = build_bank([pos], [neg], k=5, mcs=5)

    dominated = unit(np.array([0.6, 0.8, 0.0]))  # s_pos=0.6 < s_neg=0.8
    free = unit(np.array([0.8, 0.6, 0.0]))       # s_pos=0.8 > s_neg=0.6
    assert bank.max_similarity(dominated) == pytest.approx(2 * 0.6 - 0.8)  # 0.40
    assert bank.max_similarity(free) == pytest.approx(0.8)

    # Band-trip: a negative seed tilted toward the positive makes a
    # core-band lookalike (s_pos ~ 0.86) dominated (s_neg ~ 0.99); the veto
    # drags it out of the auto-tag band into the boundary band.
    close_neg = unit(np.array([1.0, 0.45, 0.0]))
    lookalike_bank = build_bank([pos], [close_neg], k=5, mcs=5)
    lookalike = unit(np.array([1.0, 0.6, 0.0]))
    s_pos = float((lookalike_bank.prototypes @ lookalike).max())
    s_neg = float(close_neg @ lookalike)
    assert s_pos >= 0.82, "without the veto this would auto-tag as core"
    assert s_neg > s_pos, "test geometry must produce a dominated lookalike"
    eff = lookalike_bank.max_similarity(lookalike)
    assert eff == pytest.approx(2 * s_pos - s_neg)
    assert eff < 0.82, "veto must demote this lookalike below tau_core"
    assert eff >= 0.68, "and into the boundary review band, not rejection"


def test_prototype_bank_veto_monotone():
    q = unit(np.array([0.8, 0.6, 0.0]))
    u = unit(np.array([0.6, -0.8, 0.0]))  # orthogonal to q

    # (a) stronger negatives never increase the effective score
    effs_weakening_negatives = []
    for theta in np.linspace(0.0, np.pi / 2, 8):
        neg_candidate = np.cos(theta) * q + np.sin(theta) * u
        bank = build_bank([np.array([1.0, 0.0, 0.0])], [neg_candidate])
        effs_weakening_negatives.append(bank.max_similarity(q))
    assert all(
        b >= a - 1e-12
        for a, b in zip(effs_weakening_negatives, effs_weakening_negatives[1:])
    )

    # (b) stronger prototype similarity never decreases the effective score
    effs_closer_prototype = []
    for phi in np.linspace(np.pi / 2, 0.0, 8):
        proto = np.cos(phi) * q + np.sin(phi) * u
        bank = build_bank([proto], [np.array([0.0, 1.0, 0.0])])
        effs_closer_prototype.append(bank.max_similarity(q))
    assert all(
        b >= a - 1e-12
        for a, b in zip(effs_closer_prototype, effs_closer_prototype[1:])
    )


def test_prototype_bank_guards():
    bank = PrototypeBank(k_neighbors=5, min_cluster_size=5)
    with pytest.raises(ValueError, match="no must-link seeds"):
        bank.fit_subclusters()
    bank.add_must_link(unit([1.0, 0.0, 0.0]))
    with pytest.raises(RuntimeError, match="fit_subclusters"):
        bank.max_similarity(unit([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="min_cluster_size must be >= 2"):
        PrototypeBank(k_neighbors=5, min_cluster_size=1)


# ---------------------------------------------------------------------------
# assign_two_tier
# ---------------------------------------------------------------------------


def test_assign_two_tier_exact_bands():
    params = make_params(tau_core=0.82, tau_boundary=0.68)
    decisions = assign_two_tier(
        ["high-core", "exact-core", "just-below-boundary", "exact-boundary",
         "just-above-reject", "far-negative"],
        [0.95, 0.82, 0.6799, 0.68, 0.679, -0.5],
        params,
    )
    tiers = [d["tier"] for d in decisions]
    assert tiers == ["core", "core", "reject", "boundary", "reject", "reject"]
    assert [d["score"] for d in decisions] == pytest.approx(
        [0.95, 0.82, 0.6799, 0.68, 0.679, -0.5]
    )
    assert all(d["reasons"] for d in decisions)
    assert set(decisions[0]) == {"instance_id", "score", "tier", "reasons"}


def test_assign_two_tier_rejects_bad_input():
    params = make_params()
    with pytest.raises(ValueError, match="non-finite"):
        assign_two_tier(["a"], [float("nan")], params)
    with pytest.raises(ValueError, match="2 instance ids but 1 scores"):
        assign_two_tier(["a", "b"], [0.5], params)
    with pytest.raises(ValueError, match="tau_boundary <= tau_core"):
        assign_two_tier(["a"], [0.5], make_params(tau_core=0.5, tau_boundary=0.6))


# ---------------------------------------------------------------------------
# Same-Panel Cannot-Link
# ---------------------------------------------------------------------------


def _decision(instance_id, score, tier="core"):
    return {"instance_id": instance_id, "score": score, "tier": tier, "reasons": []}


def test_panel_constraints_demote_runner_up_core():
    params = make_params(panel_instance_iou=0.15)
    decisions = [_decision("winner", 0.90), _decision("loser", 0.84)]
    result = apply_panel_constraints(decisions, ["same", "same"], params)

    assert [d["tier"] for d in result] == ["core", "boundary"]
    assert any("same_panel_cannot_link" in r for r in result[1]["reasons"])
    assert "winner" in result[1]["reasons"][-1]
    # purity: the caller's dicts are untouched
    assert decisions[1]["tier"] == "core" and decisions[1]["reasons"] == []


def test_panel_constraints_distinct_panels_unaffected():
    params = make_params()
    decisions = [_decision("a", 0.90), _decision("b", 0.84)]
    result = apply_panel_constraints(decisions, ["p1", "p2"], params)
    assert [d["tier"] for d in result] == ["core", "core"]


def test_panel_constraints_doppelganger_bypass():
    params = make_params()
    decisions = [_decision("a", 0.90), _decision("b", 0.84)]
    result = apply_panel_constraints(
        decisions, ["same", "same"], params, allow_doppelganger=True
    )
    assert [d["tier"] for d in result] == ["core", "core"]


def test_panel_constraints_iou_pairs_distinctness():
    """Heavily overlapping instances are one physical character and are exempt;
    spatially distinct ones (low IoU or absent pairs) get suppressed."""
    params = make_params(panel_instance_iou=0.15)
    overlapping = [_decision("dup-a", 0.91), _decision("dup-b", 0.86)]
    result = apply_panel_constraints(
        overlapping, ["same", "same"], params, iou_pairs={(0, 1): 0.90}
    )
    assert [d["tier"] for d in result] == ["core", "core"]

    distinct = [_decision("chr-a", 0.91), _decision("chr-b", 0.86)]
    result = apply_panel_constraints(
        distinct, ["same", "same"], params, iou_pairs={(0, 1): 0.05}
    )
    assert [d["tier"] for d in result] == ["core", "boundary"]

    # IoU exactly at the threshold counts as the same physical character.
    at_threshold = [_decision("thr-a", 0.91), _decision("thr-b", 0.86)]
    result = apply_panel_constraints(
        at_threshold, ["same", "same"], params, iou_pairs={(0, 1): 0.15}
    )
    assert [d["tier"] for d in result] == ["core", "core"]


def test_panel_constraints_iou_validation_errors():
    params = make_params()
    decisions = [_decision("a", 0.9), _decision("b", 0.8)]
    with pytest.raises(ValueError, match="ascending position tuples"):
        apply_panel_constraints(decisions, ["p", "p"], params, iou_pairs={(1, 0): 0.5})
    with pytest.raises(ValueError, match="IoU in"):
        apply_panel_constraints(decisions, ["p", "p"], params, iou_pairs={(0, 1): 1.5})


def test_panel_constraints_rejects_malformed_decisions():
    params = make_params()
    with pytest.raises(ValueError, match="missing key"):
        apply_panel_constraints([{"instance_id": "x", "score": 0.9}], ["p"], params)
    with pytest.raises(ValueError, match="but 2 panel ids"):
        apply_panel_constraints([_decision("a", 0.9)], ["p", "q"], params)


# ---------------------------------------------------------------------------
# Same-Page Prior Boost
# ---------------------------------------------------------------------------


def test_page_priors_boost_only_on_pages_with_core():
    decisions = [
        _decision("core", 0.90, "core"),
        _decision("boost-me", 0.60, "boundary"),
        _decision("capped", 0.99, "boundary"),
        _decision("other-page", 0.70, "boundary"),
    ]
    pages = ["p1", "p1", "p1", "p2"]
    result = apply_page_priors(decisions, pages, boost=0.06)

    assert result[0]["score"] == pytest.approx(0.90)  # core is never boosted
    assert result[1]["score"] == pytest.approx(0.66)
    assert result[2]["score"] == pytest.approx(1.0)   # capped at 1.0
    assert result[3]["score"] == pytest.approx(0.70)  # no core on p2
    assert any("page_prior_boost" in r for r in result[1]["reasons"])
    assert not any("page_prior_boost" in r for r in result[0]["reasons"])
    # purity
    assert decisions[1]["score"] == 0.60 and decisions[1]["reasons"] == []


def test_page_priors_reject_negative_boost_and_bad_lengths():
    with pytest.raises(ValueError, match="finite value >= 0"):
        apply_page_priors([_decision("a", 0.5)], ["p"], boost=-0.1)
    with pytest.raises(ValueError, match="but 2 page ids"):
        apply_page_priors([_decision("a", 0.5)], ["p", "q"], boost=0.06)


# ---------------------------------------------------------------------------
# assign_izutsumi — end-to-end two-pass flow and call order
# ---------------------------------------------------------------------------


class _RecordingBank:
    """Bank stub: fixed per-instance scores, records every scoring call."""

    def __init__(self, scores):
        self.scores = list(scores)
        self.calls: list[str] = []

    def max_similarity(self, embedding) -> float:
        self.calls.append("bank.max_similarity")
        return self.scores.pop(0)


def test_assign_izutsumi_two_pass_order_and_context(monkeypatch):
    calls: list[str] = []
    originals = {
        name: getattr(clustering, name)
        for name in (
            "assign_two_tier",
            "apply_page_priors",
            "apply_panel_constraints",
            "subcluster_embeddings",
        )
    }
    for name, original in originals.items():
        def wrap(*args, __name=name, __original=original, **kwargs):
            calls.append(__name)
            return __original(*args, **kwargs)
        monkeypatch.setattr(clustering, name, wrap)

    params = make_params(k_neighbors=2, min_cluster_size=2)
    bank = _RecordingBank([0.90, 0.85, 0.65, 0.90])
    e_i, e_j = unit([1.0, 0.0, 0.0]), unit([0.0, 1.0, 0.0])
    records = [
        {"instance_id": "char-a", "page": "p1", "panel": "p1", "embedding": e_i, "crop_path": "crops/a.jpg"},
        {"instance_id": "char-b", "page": "p1", "panel": "p1", "embedding": e_i, "crop_path": "crops/b.jpg"},
        {"instance_id": "back-view", "page": "p1", "panel": "p1-adjacent", "embedding": e_i, "crop_path": "crops/c.jpg"},
        {"instance_id": "other-page", "page": "p2", "panel": "p2", "embedding": e_j, "crop_path": "crops/d.jpg"},
    ]

    decisions = assign_izutsumi(records, bank, params)

    # exact call order of the two-pass assignment
    assert calls == [
        "assign_two_tier",       # pass 1: detect cores
        "apply_page_priors",     # boost non-cores on core pages
        "assign_two_tier",       # final tier split on boosted scores
        "apply_panel_constraints",  # cannot-link demotion wins over priors
        "subcluster_embeddings",    # pure gallery-grouping annotation
    ]
    assert bank.calls == ["bank.max_similarity"] * 4

    by_id = {d["instance_id"]: d for d in decisions}
    # char-a wins its panel; char-b was core and gets demoted with a reason
    assert by_id["char-a"]["tier"] == "core"
    assert by_id["char-b"]["tier"] == "boundary"
    assert any("same_panel_cannot_link" in r for r in by_id["char-b"]["reasons"])
    # back-view: reject -> boosted to boundary by the same-page prior
    assert by_id["back-view"]["tier"] == "boundary"
    assert any("page_prior_boost" in r for r in by_id["back-view"]["reasons"])
    assert by_id["back-view"]["score"] == pytest.approx(0.65 + 0.06)
    # the other page's core is untouched
    assert by_id["other-page"]["tier"] == "core"
    # crop passthrough + integer subcluster annotation
    assert by_id["char-a"]["crop_path"] == "crops/a.jpg"
    assert all(isinstance(d["subcluster"], int) for d in decisions)
    # the three near-identical layouts share a sub-cluster, the orthogonal
    # page does not
    assert by_id["char-a"]["subcluster"] == by_id["char-b"]["subcluster"]
    assert by_id["other-page"]["subcluster"] != by_id["char-a"]["subcluster"]
    # pass-1 reasons are preserved beneath the final band rationale
    assert any("0.7100" in r or "0.71" in r for r in by_id["back-view"]["reasons"])


def test_assign_izutsumi_validates_and_handles_empty(monkeypatch):
    params = make_params()
    bank = build_bank([np.array([1.0, 0.0, 0.0])], [], k=2, mcs=2)
    assert assign_izutsumi([], bank, params) == []
    records = [{
        "instance_id": "x", "page": "p", "embedding": unit([1.0, 0.0]),
        "crop_path": "c.jpg",
    }]
    with pytest.raises(ValueError, match="missing key.*panel"):
        assign_izutsumi(records, bank, params)


def test_assign_izutsumi_end_to_end_real_bank_semantics():
    """Full pipeline without mocks: scores flow through priors and panels."""
    params = make_params(k_neighbors=2, min_cluster_size=2)
    bank = _RecordingBank([0.90, 0.60, 0.60, 0.60])
    # pages alternate p0/p1; c0 (i=0) and c2 (i=2) land on page p0, which
    # holds the only core; c1 additionally shares c0's panel.
    records = [
        {"instance_id": f"c{i}", "page": f"p{i % 2}", "panel": f"p{i % 2}",
         "embedding": unit([1.0, 0.0, 0.0]), "crop_path": f"crops/{i}.jpg"}
        for i in range(4)
    ]
    records[1]["panel"] = "p0"
    decisions = assign_izutsumi(records, bank, params)
    by_id = {d["instance_id"]: d for d in decisions}

    # c0 wins the panel and the page as the only core.
    assert by_id["c0"]["tier"] == "core"
    # c1: reject in pass 1, on page p1 (no core there) -> unboosted reject;
    # sharing c0's panel cannot demote a non-core any further.
    assert by_id["c1"]["score"] == pytest.approx(0.60)
    assert by_id["c1"]["tier"] == "reject"
    # c2: reject in pass 1, but on page p0 -> page prior lifts it to 0.66,
    # still below tau_boundary (0.68), so it stays rejected. Priors never
    # mint tiers across the band edge.
    assert by_id["c2"]["score"] == pytest.approx(0.66)
    assert by_id["c2"]["tier"] == "reject"
    assert any("page_prior_boost" in r for r in by_id["c2"]["reasons"])
    # c3: reject, own page without a core, unboosted.
    assert by_id["c3"]["score"] == pytest.approx(0.60)
    assert by_id["c3"]["tier"] == "reject"


# ---------------------------------------------------------------------------
# build_review_gallery
# ---------------------------------------------------------------------------


def test_build_review_gallery_groups_by_subcluster(tmp_path):
    crops_dir = tmp_path / "crops"
    crops_dir.mkdir()
    (crops_dir / "b1.jpg").write_bytes(b"fake-jpg")
    (crops_dir / "b2.png").write_bytes(b"fake-png")
    out_html = tmp_path / "out" / "boundary_review.html"

    decisions = [
        {"instance_id": "core-1", "score": 0.95, "tier": "core", "reasons": [], "subcluster": 0, "crop_path": "crops/x.jpg"},
        {"instance_id": "b-lo", "score": 0.796, "tier": "boundary", "reasons": [], "subcluster": 0, "crop_path": "crops/b1.jpg"},
        {"instance_id": "b-hi", "score": 0.81, "tier": "boundary", "reasons": [], "subcluster": 0, "crop_path": "crops/b1.jpg"},
        {"instance_id": "b-new", "score": 0.70, "tier": "boundary", "reasons": [], "subcluster": 1, "crop_path": "crops/b2.png"},
        {"instance_id": "b-loose", "score": 0.69, "tier": "boundary", "reasons": [], "subcluster": -1, "crop_path": None},
        {"instance_id": "rej-1", "score": 0.10, "tier": "reject", "reasons": [], "subcluster": -1, "crop_path": "crops/y.jpg"},
    ]
    crops = {"b-lo": crops_dir / "b1.jpg", "b-new": crops_dir / "b2.png"}

    result = build_review_gallery(decisions, crops, out_html)

    assert result == out_html
    html = out_html.read_text(encoding="utf-8")
    # boundary only: core and reject ids never leak in
    assert "core-1" not in html and "rej-1" not in html
    # sections in subcluster order, with noise grouped last
    assert html.index("Subcluster 0") < html.index("Subcluster 1") < html.index("Ungrouped")
    assert "2 instance(s)" in html and "1 instance(s)" in html
    # relative image paths, no JS anywhere
    assert "src='../crops/b1.jpg'" in html
    assert "<script" not in html
    # per-crop scores rendered, boundary id missing from crops -> placeholder
    assert "0.7960" in html and "0.8100" in html
    assert "crop unavailable" in html
    # empty gallery case
    empty = build_review_gallery(decisions[:1], {}, tmp_path / "empty.html")
    assert "No boundary instances to review." in empty.read_text(encoding="utf-8")


def test_assign_izutsumi_bbox_iou_feeds_panel_distinctness():
    """Record bboxes reach Same-Panel Cannot-Link (review M2 regression).

    Heavily overlapping same-panel instance boxes are one physical
    detection chain and are exempt from mutual suppression; distinct
    boxes (IoU < panel_instance_iou) and bbox-less records keep the
    historical single-core demotion.
    """
    params = make_params(k_neighbors=2, min_cluster_size=2)
    embedding = unit([1.0, 0.0, 0.0])
    shared = {"page": "p1", "panel": "p1", "embedding": embedding}

    overlapping = [
        {**shared, "instance_id": "a", "crop_path": "crops/a.jpg", "bbox": [10, 10, 200, 400]},
        {**shared, "instance_id": "b", "crop_path": "crops/b.jpg", "bbox": [16, 12, 206, 402]},
    ]
    decisions = assign_izutsumi(overlapping, _RecordingBank([0.90, 0.88]), params)
    by_id = {d["instance_id"]: d for d in decisions}
    assert by_id["a"]["tier"] == "core"
    assert by_id["b"]["tier"] == "core"  # same physical chain: exempt

    distinct = [
        {**shared, "instance_id": "a", "crop_path": "crops/a.jpg", "bbox": [10, 10, 100, 200]},
        {**shared, "instance_id": "b", "crop_path": "crops/b.jpg", "bbox": [300, 300, 500, 600]},
    ]
    decisions = assign_izutsumi(distinct, _RecordingBank([0.90, 0.88]), params)
    by_id = {d["instance_id"]: d for d in decisions}
    assert by_id["b"]["tier"] == "boundary"
    assert any("same_panel_cannot_link" in r for r in by_id["b"]["reasons"])

    bboxless = [
        {**shared, "instance_id": "a", "crop_path": "crops/a.jpg"},
        {**shared, "instance_id": "b", "crop_path": "crops/b.jpg"},
    ]
    decisions = assign_izutsumi(bboxless, _RecordingBank([0.90, 0.88]), params)
    by_id = {d["instance_id"]: d for d in decisions}
    assert by_id["b"]["tier"] == "boundary"  # distinct-by-default fallback


def test_assign_izutsumi_page_boost_cannot_displace_natural_core():
    """Review B2 regression: a lookalike lifted past tau_core by the page
    prior must never outrank a natural core in the Same-Panel ranking.

    True core at raw 0.84; lookalike at raw 0.80 on the same page gets
    boosted to 0.86 (core) by the prior — the panel rule still elects the
    natural core and demotes the boost-promoted twin to boundary.
    """
    params = make_params(k_neighbors=2, min_cluster_size=2)
    embedding = unit([1.0, 0.0, 0.0])
    records = [
        {"instance_id": "truesi", "page": "p1", "panel": "p1", "embedding": embedding, "crop_path": "crops/a.jpg"},
        {"instance_id": "lookalike", "page": "p1", "panel": "p1", "embedding": embedding, "crop_path": "crops/b.jpg"},
    ]

    decisions = assign_izutsumi(records, _RecordingBank([0.84, 0.80]), params)
    by_id = {d["instance_id"]: d for d in decisions}

    assert by_id["truesi"]["tier"] == "core"
    assert by_id["lookalike"]["tier"] == "boundary"
    assert by_id["lookalike"]["score"] == pytest.approx(0.86)  # boosted... but demoted
    assert by_id["lookalike"]["raw_score"] == pytest.approx(0.80)
    assert any("same_panel_cannot_link" in r for r in by_id["lookalike"]["reasons"])
