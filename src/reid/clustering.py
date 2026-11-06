"""Semi-supervised multi-prototype clustering for Stage 2 Izutsumi re-identification.

Pipeline (plan task 5.3 / two-tier decision output):

1. ``knn_graph`` builds an exact k-NN cosine similarity graph over L2-normalized
   character-instance embeddings. ``faiss`` is used opportunistically when
   installed; otherwise an exact numpy fallback runs. The two are equivalent:
   an ``IndexFlatIP`` over L2-normalized vectors *is* a cosine-similarity
   search, and its top-k is the same top-k a matmul + argpartition produces
   (the numpy path may break similarity ties differently than faiss, but
   never changes the neighbor set for distinct similarities).
2. ``subcluster_embeddings`` partitions instances with HDBSCAN
   (sklearn.cluster.HDBSCAN, ``metric="precomputed"``) over an exact dense
   cosine-distance matrix by default (documented memory note, capped at
   :data:`MAX_SUBCLUSTER_INSTANCES`), or — opt-in — over a sparse
   precomputed k-NN cosine-distance graph for memory-lean bulk grouping;
   the sparse graph is bridged to one connected component exactly per
   sklearn's documented sparse-graph conventions.
3. ``PrototypeBank`` turns verified must-link seeds (Izutsumi crops) into
   sub-cluster medoid prototypes (standard / chibi / hooded / body / fanart
   modes) and applies a documented cannot-link veto against negative seeds
   (Laios, Marcille, ...).
4. Two-tier decisions: core instances (score >= ``tau_core``) are auto-tagged;
   boundary instances (``[tau_boundary, tau_core)``) are grouped per
   sub-cluster into an HTML review gallery; the rest are rejected.
   Contextual rules: *Same-Panel Cannot-Link* (at most one spatially distinct
   core instance per panel) and *Same-Page Prior Boost* (a page containing a
   core boosts adjacent non-core candidates, capped at 1.0). Constraints win
   over priors: panel suppression runs after the final tier split.

Record contract for :func:`assign_izutsumi`: each record is a dict with keys
``instance_id``, ``page``, ``panel``, ``embedding`` (1-D array-like) and
``crop_path``. Without panel segmentation, ``page`` and ``panel`` are both the
source image path, which makes every image its own panel; multi-character
images then naturally compete under Same-Panel Cannot-Link.
"""

from __future__ import annotations

import html
import logging
import os
from pathlib import Path
from urllib.parse import quote as _url_quote

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from src.config import ClusteringParams

logger = logging.getLogger(__name__)

# HDBSCAN clustering is only bounded up to this many instances: the default
# dense precomputed distance matrix costs 8 * n**2 bytes (~3.2 GB at
# n = 20_000). Larger inputs fail loudly instead of dying to an unexplained
# OOM later.
MAX_SUBCLUSTER_INSTANCES = 20_000

# scipy's minimum_spanning_tree treats stored zeros as absent edges, so a
# cosine distance of exactly 0.0 (identical embeddings) would silently drop
# the edge and make the graph look disconnected. All stored distances are
# floored to this value: a monotone transform that only affects duplicates.
_MIN_DISTANCE = 1e-12

# Row block size for chunked similarity matmuls (memory ceiling per block).
_ROW_BLOCK = 2048

TIER_CORE = "core"
TIER_BOUNDARY = "boundary"
TIER_REJECT = "reject"

# The sklearn path requires every sparse row to carry at least min_samples
# (= min_cluster_size) stored neighbor distances; k_neighbors guarantees that.
_ASSIGN_RECORD_KEYS = ("instance_id", "page", "panel", "embedding", "crop_path")
_DECISION_KEYS = ("instance_id", "score", "tier")


def _require_decision_keys(decisions: list[dict]) -> None:
    """Fail fast on malformed decision dicts before any rule consumes them."""
    for position, decision in enumerate(decisions):
        missing = [key for key in _DECISION_KEYS if key not in decision]
        if missing:
            raise ValueError(
                f"decision at position {position} is missing key(s) {missing}"
            )


# ---------------------------------------------------------------------------
# Embedding hygiene (Law 2 — parse into trusted state at the boundary)
# ---------------------------------------------------------------------------


def _as_embedding_matrix(embeddings) -> np.ndarray:
    """Return a validated, L2-normalized float64 copy of ``embeddings``."""
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(
            f"embeddings must be a non-empty 2-D array, got shape {matrix.shape}"
        )
    bad_rows = np.flatnonzero(~np.isfinite(matrix).all(axis=1))
    if bad_rows.size:
        raise ValueError(
            f"embeddings contain NaN/inf values at row(s) {bad_rows.tolist()}"
        )
    return _l2_normalize_matrix(matrix)


def _as_embedding_vector(embedding) -> np.ndarray:
    """Return a validated, L2-normalized float64 copy of a single embedding."""
    vector = np.asarray(embedding, dtype=np.float64).reshape(-1)
    if vector.size == 0:
        raise ValueError("embedding must be a non-empty 1-D vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError("embedding contains NaN/inf values")
    return _l2_normalize_matrix(vector.reshape(1, -1))[0]


def _l2_normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """Row-normalize float64 ``matrix``; zero rows are an invalid state."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    zero_rows = np.flatnonzero(norms[:, 0] <= 0.0)
    if zero_rows.size:
        raise ValueError(
            f"cannot L2-normalize zero-norm embedding row(s) {zero_rows.tolist()}"
        )
    return matrix / norms


def l2_normalize(embeddings) -> np.ndarray:
    """L2-normalize a single embedding (1-D) or a batch (2-D), float64 out.

    Exposed for callers that persist or fuse embeddings (e.g. the CLI) and
    need the same trusted-state guarantee without importing private helpers.
    """
    array = np.asarray(embeddings, dtype=np.float64)
    if array.ndim == 1:
        return _as_embedding_vector(array)
    if array.ndim == 2:
        return _as_embedding_matrix(array)
    raise ValueError(f"expected a 1-D or 2-D array, got shape {array.shape}")


# ---------------------------------------------------------------------------
# k-NN graph (plan: FAISS IndexFlatIP, k = 25 — exact numpy fallback)
# ---------------------------------------------------------------------------


def knn_graph(embeddings, k: int, use_faiss_if_available: bool = True) -> np.ndarray:
    """Return the k nearest-neighbor indices per row, most similar first.

    Self is never a neighbor. Rows with fewer than ``k`` valid neighbors
    (small inputs) return a clipped graph with a warning. Ties are broken
    deterministically by ascending neighbor index.
    """
    indices, _ = _knn_pairs(embeddings, k, use_faiss_if_available)
    return indices


def _knn_pairs(embeddings, k: int, use_faiss_if_available: bool = True):
    """Return (neighbor indices [n, k_eff], similarities [n, k_eff])."""
    matrix = _as_embedding_matrix(embeddings)
    n = matrix.shape[0]
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    k_eff = min(k, n - 1)
    if k_eff < k:
        if k_eff < 1:
            logger.warning("knn graph on %d instance(s): no neighbors to link", n)
        else:
            logger.warning(
                "k=%d clipped to n-1=%d (%d instances less than k+1)",
                k, k_eff, n,
            )
    if k_eff < 1:
        empty_i = np.empty((n, 0), dtype=np.int64)
        empty_s = np.empty((n, 0), dtype=np.float64)
        return empty_i, empty_s

    if use_faiss_if_available:
        faiss = _try_import_faiss()
        if faiss is not None:
            return _knn_pairs_faiss(faiss, matrix, k_eff)
    return _knn_pairs_numpy(matrix, k_eff)


def _try_import_faiss():
    """Return the faiss module if importable, else None (never a hard dep)."""
    try:
        import faiss  # noqa: PLC0415 — optional acceleration, guarded
    except Exception:  # pragma: no cover — faiss is not installed locally
        logger.debug("faiss unavailable; using exact numpy k-NN (documented equivalent)")
        return None
    return faiss


def _knn_pairs_faiss(faiss, matrix: np.ndarray, k_eff: int):
    """IndexFlatIP over L2-normalized vectors == top-k cosine similarities."""
    vectors = matrix.astype(np.float32)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    # k_eff + 1 candidates so the row's own membership can be dropped below.
    similarities, searched = index.search(vectors, k_eff + 1)
    n = matrix.shape[0]
    is_self = searched == np.arange(n)[:, None]
    if not is_self.any(axis=1).all():  # pragma: no cover — defensive
        raise RuntimeError("faiss failed to return each query's own row")
    # Stable-sort self hits last per row, then keep the first k_eff columns.
    order = np.argsort(is_self, axis=1, kind="stable")
    neighbor_idx = np.take_along_axis(searched, order, axis=1)[:, :k_eff]
    neighbor_sims = np.take_along_axis(similarities, order, axis=1)[:, :k_eff]

    # Same deterministic tie-break as the numpy path.
    by_index = np.argsort(neighbor_idx, axis=1, kind="stable")
    neighbor_idx = np.take_along_axis(neighbor_idx, by_index, axis=1)
    neighbor_sims = np.take_along_axis(neighbor_sims, by_index, axis=1)
    by_sim = np.argsort(-neighbor_sims, axis=1, kind="stable")
    return (
        np.take_along_axis(neighbor_idx, by_sim, axis=1).astype(np.int64),
        np.take_along_axis(neighbor_sims, by_sim, axis=1).astype(np.float64),
    )


def _knn_pairs_numpy(matrix: np.ndarray, k_eff: int):
    """Chunked exact top-k cosine: block matmul + argpartition per row."""
    n = matrix.shape[0]
    indices = np.empty((n, k_eff), dtype=np.int64)
    similarities = np.empty((n, k_eff), dtype=np.float64)
    for start in range(0, n, _ROW_BLOCK):
        stop = min(start + _ROW_BLOCK, n)
        block_sims = matrix[start:stop] @ matrix.T
        # Self-similarity is excluded from the candidate pool per row.
        local = np.arange(stop - start)
        block_sims[local, start + local] = -np.inf

        candidate_idx = np.argpartition(block_sims, -k_eff, axis=1)[:, -k_eff:]
        candidate_sims = np.take_along_axis(block_sims, candidate_idx, axis=1)
        # Deterministic tie-break: order candidates by neighbor index first,
        # then stable-sort by similarity so ties keep ascending index order.
        by_index = np.argsort(candidate_idx, axis=1, kind="stable")
        candidate_idx = np.take_along_axis(candidate_idx, by_index, axis=1)
        candidate_sims = np.take_along_axis(candidate_sims, by_index, axis=1)
        by_sim = np.argsort(-candidate_sims, axis=1, kind="stable")
        indices[start:stop] = np.take_along_axis(candidate_idx, by_sim, axis=1)
        similarities[start:stop] = np.take_along_axis(candidate_sims, by_sim, axis=1)
    return indices, similarities


# ---------------------------------------------------------------------------
# HDBSCAN density sub-clustering (sparse precomputed k-NN graph)
# ---------------------------------------------------------------------------


def subcluster_embeddings(
    embeddings,
    k_neighbors: int = 25,
    min_cluster_size: int = 5,
    use_sparse_knn_graph: bool = False,
) -> np.ndarray:
    """HDBSCAN labels (``-1`` = noise / outlier) over cosine distances.

    Default path: an exact precomputed dense cosine-distance matrix.
    Faithful noise labels need the outlier-to-outlier distances, which a
    k-NN cutoff can silently drop (a lone point whose nearest stored edges
    are all cluster members gets absorbed into that cluster instead of
    being declared noise). The dense matrix costs ``8 * n**2`` bytes
    (~3.2 GB at n = 20,000), built in row blocks to bound the working set;
    above :data:`MAX_SUBCLUSTER_INSTANCES` this fails loudly instead of
    risking an OOM.

    Opt-in sparse path (``use_sparse_knn_graph=True``): HDBSCAN over a
    sparse precomputed k-NN cosine-distance graph, following
    sklearn.cluster.HDBSCAN's sparse conventions exactly — CSR, symmetric,
    off-diagonal only (a stored zero would read as a missing edge), and
    >= ``min_samples`` stored distances per row. Disconnected k-NN graphs
    — the normal shape of well-separated blobs — are joined with their
    minimal true cross-component distances, which are exactly the edges a
    dense-matrix HDBSCAN puts at the top of its minimum spanning tree, so
    the bridging invents no cluster structure. Use it for memory-lean bulk
    gallery grouping when label fidelity at the noise seam is acceptable.

    sklearn's default eom ("excess of mass") selection is kept: leaf
    selection fragments smooth density continua into fine leaves. eom's
    dropped noise seeds still get covered — here and inside
    :class:`PrototypeBank` — by verified-seed fallbacks rather than lost.

    Degenerate inputs (fewer instances than ``min_cluster_size``) return
    all-noise labels with a warning: that is a valid empty result, not an
    error state.
    """
    matrix = _as_embedding_matrix(embeddings)
    n = matrix.shape[0]
    if min_cluster_size < 2:
        raise ValueError(f"min_cluster_size must be >= 2, got {min_cluster_size}")
    if not use_sparse_knn_graph and n > MAX_SUBCLUSTER_INSTANCES:
        raise ValueError(
            f"{n} instances exceed the dense clustering budget of "
            f"{MAX_SUBCLUSTER_INSTANCES} (the precomputed distance matrix "
            f"would need ~{8 * n * n / 1e9:.1f} GB). Sample the instance set, "
            "pass use_sparse_knn_graph=True for memory-lean k-NN grouping, or "
            "profile memory before raising clustering.MAX_SUBCLUSTER_INSTANCES."
        )

    if n < min_cluster_size:
        logger.warning(
            "subclustering %d instance(s) with min_cluster_size=%d: "
            "no density-supported cluster can exist, labeling all as noise",
            n, min_cluster_size,
        )
        return np.full(n, -1, dtype=np.int64)

    if use_sparse_knn_graph:
        if k_neighbors < min_cluster_size:
            raise ValueError(
                f"k_neighbors={k_neighbors} must be >= min_cluster_size="
                f"{min_cluster_size}: the sparse graph must store at least "
                "min_samples neighbor distances per row"
            )
        indices, similarities = _knn_pairs(matrix, k_neighbors, use_faiss_if_available=True)
        graph = _connected_knn_graph(matrix, indices, similarities)
    else:
        if n > MAX_SUBCLUSTER_INSTANCES // 4:
            logger.info(
                "dense precomputed distance matrix for %d instances "
                "(~%.1f GB); this is the documented dense-path memory note",
                n, 8 * n * n / 1e9,
            )
        graph = _dense_cosine_distances(matrix)

    # Deferred import: keep knn-only users (and tests) free of the sklearn
    # import cost.
    from sklearn.cluster import HDBSCAN  # noqa: PLC0415 — lazy heavy import

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="precomputed",
        allow_single_cluster=True,  # a unimodal set is ONE sub-cluster, not noise
        copy=False,  # the graph is module-owned scratch; mutating it is fine
    )
    clusterer.fit(graph)
    labels = np.asarray(clusterer.labels_, dtype=np.int64)
    n_clusters = len(set(labels.tolist()) - {-1})
    logger.info(
        "HDBSCAN subclustering: %d instances -> %d sub-cluster(s), %d noise",
        n, n_clusters, int((labels == -1).sum()),
    )
    return labels


def _dense_cosine_distances(matrix: np.ndarray) -> np.ndarray:
    """Exact symmetric cosine distances (zero diagonal), built in blocks.

    Block-wise preparation keeps the working set at ``_ROW_BLOCK * n * 8``
    bytes on top of the matrix itself; the per-element asymmetry between
    block matmuls (last-bit float ordering) is far inside the symmetry
    tolerance of sklearn's precomputed-matrix validation.
    """
    n = matrix.shape[0]
    distances = np.empty((n, n), dtype=np.float64)
    for start in range(0, n, _ROW_BLOCK):
        stop = min(start + _ROW_BLOCK, n)
        distances[start:stop] = 1.0 - (matrix[start:stop] @ matrix.T)
    np.fill_diagonal(distances, 0.0)
    return distances


def _connected_knn_graph(matrix: np.ndarray, indices: np.ndarray, similarities: np.ndarray) -> csr_matrix:
    """Build the symmetric bridged k-NN cosine-distance CSR graph."""
    n, k = indices.shape
    rows = np.repeat(np.arange(n, dtype=np.int64), k)
    cols = indices.reshape(-1)
    # Cosine distance from cosine similarity; floored so identical
    # embeddings do not produce stored zeros (dropped by scipy's MST).
    dists = np.maximum(1.0 - similarities.reshape(-1), _MIN_DISTANCE)
    directed = csr_matrix((dists, (rows, cols)), shape=(n, n))
    # Directed k-NN lists of a symmetric metric hold identical values in both
    # directions (<= k neighbors per row, so no duplicate cells exist), so the
    # union with clamped maximum collapses to a symmetric graph with each
    # stored distance appearing exactly once.
    graph = directed.maximum(directed.T)
    graph.data = np.maximum(graph.data, _MIN_DISTANCE)

    n_components, component_labels = connected_components(graph, directed=False)
    if n_components <= 1:
        return graph

    bridge_rows, bridge_cols, bridge_dists = _bridging_edges(matrix, component_labels)
    bridge = csr_matrix(
        (bridge_dists, (bridge_rows, bridge_cols)), shape=(n, n)
    )
    graph = graph.maximum(bridge)
    logger.info(
        "k-NN graph had %d components; added %d bridging edge(s)",
        n_components, bridge_rows.size // 2,
    )
    return graph


def _bridging_edges(matrix: np.ndarray, component_labels: np.ndarray):
    """Minimal-distance cross-component edges joining all components.

    For every instance, the nearest instance of a *different* component is
    found (chunked exact pass over cosine similarities); a union-find then
    keeps only the cheapest edge per component pair until the whole graph
    is connected. These are true pairwise distances, not placeholders: the
    dense HDBSCAN minimum spanning tree contains exactly these edges.
    """
    n = matrix.shape[0]
    best_col = np.full(n, -1, dtype=np.int64)
    best_sim = np.full(n, -np.inf, dtype=np.float64)
    for start in range(0, n, _ROW_BLOCK):
        stop = min(start + _ROW_BLOCK, n)
        block_sims = matrix[start:stop] @ matrix.T
        rows_comp = component_labels[start:stop][:, None]
        cross = rows_comp != component_labels[None, :]
        block_sims = np.where(cross, block_sims, -np.inf)
        local_pick = np.argmax(block_sims, axis=1)
        best_col[start:stop] = local_pick
        best_sim[start:stop] = block_sims[np.arange(stop - start), local_pick]

    # Cheap edges first; keep an edge only when it links two components that
    # are not connected yet (union-find over component labels).
    order = np.argsort(-best_sim)
    parent = {int(label): int(label) for label in np.unique(component_labels)}

    def find(label: int) -> int:
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    rows: list[int] = []
    cols: list[int] = []
    dists: list[float] = []
    for row in order:
        dist = 1.0 - best_sim[row]
        if not np.isfinite(dist):
            continue
        col = int(best_col[row])
        left, right = find(int(component_labels[row])), find(int(component_labels[col]))
        if left == right:
            continue
        parent[left] = right
        # Store both directions so the CSR stays symmetric.
        rows.extend((row, col))
        cols.extend((col, row))
        dists.extend((max(dist, _MIN_DISTANCE), max(dist, _MIN_DISTANCE)))
    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(cols, dtype=np.int64),
        np.asarray(dists, dtype=np.float64),
    )


def medoid_of(cluster_embeddings) -> int:
    """Index of the member maximizing the summed within-cluster cosine similarity.

    For L2-normalized embeddings, summed distance is minimized exactly when
    summed similarity is maximized, so the medoid is ``argmax`` of the
    row-summed similarity matrix. Ties resolve to the lowest index.
    """
    matrix = _as_embedding_matrix(cluster_embeddings)
    n = matrix.shape[0]
    if n == 1:
        return 0
    sums = np.empty(n, dtype=np.float64)
    for start in range(0, n, _ROW_BLOCK):
        stop = min(start + _ROW_BLOCK, n)
        sums[start:stop] = (matrix[start:stop] @ matrix.T).sum(axis=1)
    return int(np.argmax(sums))


# ---------------------------------------------------------------------------
# PrototypeBank — must-link / cannot-link seeds
# ---------------------------------------------------------------------------


class PrototypeBank:
    """Multi-prototype Izutsumi matcher with a cannot-link veto.

    Positive (must-link) seeds are grouped with :func:`subcluster_embeddings`;
    each density sub-cluster contributes its medoid as a prototype, so
    Izutsumi's distinct visual modes (standard, chibi/feral, hooded,
    full-body, colored fanart) never have to collapse into one centroid.
    Seed crops that HDBSCAN marks as noise still become singleton
    prototypes: they were human-verified in Label Studio, and max-similarity
    matching only ever gains recall from extra coverage prototypes.

    Cannot-link (negative) seeds are stored raw and apply the following
    **documented veto rule** in :meth:`max_similarity`::

        s_pos = max cosine similarity to any prototype
        s_neg = max cosine similarity to any cannot-link seed
        score_eff = s_pos                        if s_neg <= s_pos
        score_eff = 2 * s_pos - s_neg            if s_neg > s_pos

    The rule is monotone (non-decreasing in ``s_pos``, non-increasing in
    ``s_neg``), continuous, and it only trips when a known negative seed
    outranks every Izutsumi prototype: the negative's margin of dominance
    is then deducted from the Izutsumi score, demoting lookalikes of
    recurring non-Izutsumi characters (Laios, Marcille, ...) out of the
    auto-tag core band even when a sloppy prototype matches them strongly.
    """

    def __init__(self, k_neighbors: int = 25, min_cluster_size: int = 5):
        k_neighbors = int(k_neighbors)
        min_cluster_size = int(min_cluster_size)
        if k_neighbors < 1:
            raise ValueError(f"k_neighbors must be >= 1, got {k_neighbors}")
        if min_cluster_size < 2:
            raise ValueError(f"min_cluster_size must be >= 2, got {min_cluster_size}")
        self._k_neighbors = k_neighbors
        self._min_cluster_size = min_cluster_size
        self._must_link: list[np.ndarray] = []
        self._cannot_link: list[np.ndarray] = []
        self._prototypes: np.ndarray | None = None

    def add_must_link(self, embedding) -> None:
        """Register a verified Izutsumi seed embedding."""
        self._prototypes = None
        self._must_link.append(_as_embedding_vector(embedding))

    def add_cannot_link(self, embedding) -> None:
        """Register a verified non-Izutsumi seed embedding."""
        self._cannot_link.append(_as_embedding_vector(embedding))

    @property
    def prototypes(self) -> np.ndarray:
        """Fitted prototype medoids, shape [n_prototypes, dim]."""
        if self._prototypes is None:
            raise RuntimeError(
                "PrototypeBank.fit_subclusters() must run before scoring"
            )
        return self._prototypes

    def fit_subclusters(self) -> None:
        """Derive prototypes from HDBSCAN sub-clusters of the must-link seeds."""
        if not self._must_link:
            raise ValueError(
                "no must-link seeds added yet: call add_must_link(embedding) first"
            )
        seeds = np.stack(self._must_link)
        labels = subcluster_embeddings(
            seeds, k_neighbors=self._k_neighbors, min_cluster_size=self._min_cluster_size
        )
        prototypes: list[np.ndarray] = []
        for label in sorted(set(labels.tolist()) - {-1}):
            members = seeds[labels == label]
            prototypes.append(members[medoid_of(members)])
        noise_seeds = seeds[labels == -1]
        prototypes.extend(noise_seeds)  # verified singleton modes

        if not prototypes:  # pragma: no cover — labels always partition the seeds
            raise RuntimeError("HDBSCAN produced neither clusters nor noise for seeds")
        self._prototypes = np.stack(prototypes)
        logger.info(
            "PrototypeBank: %d must-link seed(s) -> %d prototype(s) "
            "(%d sub-cluster medoid(s), %d singleton(s))",
            len(self._must_link), len(self._prototypes),
            len(set(labels.tolist()) - {-1}), len(noise_seeds),
        )

    def max_similarity(self, embedding) -> float:
        """Effective Izutsumi similarity of one embedding (veto rule applied)."""
        prototypes = self.prototypes
        vector = _as_embedding_vector(embedding)
        pos_scores = prototypes @ vector
        s_pos = float(pos_scores.max()) if pos_scores.size else -np.inf
        if self._cannot_link:
            neg_scores = np.stack(self._cannot_link) @ vector
            s_neg = float(neg_scores.max())
        else:
            s_neg = -np.inf
        # Documented cannot-link veto (see class docstring).
        if s_neg <= s_pos or s_neg == -np.inf:
            return s_pos
        return 2.0 * s_pos - s_neg


# ---------------------------------------------------------------------------
# Two-tier decisions & contextual constraints
# ---------------------------------------------------------------------------


def _validate_tiers(params: ClusteringParams) -> None:
    if not 0.0 < params.tau_boundary <= params.tau_core <= 1.0:
        raise ValueError(
            "clustering thresholds violate 0 < tau_boundary <= tau_core <= 1: "
            f"tau_boundary={params.tau_boundary}, tau_core={params.tau_core}"
        )


def assign_two_tier(instance_ids, scores_eff, params: ClusteringParams) -> list[dict]:
    """Map effective scores to tier decisions (pure; reasons carry the band).

    Bands per the two-tier spec: ``score >= tau_core`` -> core (auto-tag),
    ``[tau_boundary, tau_core)`` -> boundary (human review), anything below
    -> reject.
    """
    _validate_tiers(params)
    scores = np.asarray(scores_eff, dtype=np.float64).reshape(-1)
    if len(instance_ids) != scores.shape[0]:
        raise ValueError(
            f"{len(instance_ids)} instance ids but {scores.shape[0]} scores"
        )
    non_finite = np.flatnonzero(~np.isfinite(scores))
    if non_finite.size:
        raise ValueError(f"non-finite score(s) at position(s) {non_finite.tolist()}")

    decisions: list[dict] = []
    for instance_id, score in zip(instance_ids, scores):
        score = float(score)
        if score >= params.tau_core:
            tier, reason = TIER_CORE, f"score >= tau_core ({params.tau_core:.3f})"
        elif score >= params.tau_boundary:
            tier, reason = TIER_BOUNDARY, (
                f"tau_boundary <= score < tau_core "
                f"({params.tau_boundary:.3f} <= {score:.4f} < {params.tau_core:.3f})"
            )
        else:
            tier, reason = TIER_REJECT, f"score < tau_boundary ({params.tau_boundary:.3f})"
        decisions.append(
            {
                "instance_id": instance_id,
                "score": score,
                "tier": tier,
                "reasons": [reason],
            }
        )
    return decisions


def apply_panel_constraints(
    decisions: list[dict],
    panel_ids,
    params: ClusteringParams,
    iou_pairs: dict[tuple[int, int], float] | None = None,
    allow_doppelganger: bool = False,
) -> list[dict]:
    """Same-Panel Cannot-Link: at most one spatially distinct core per panel.

    ``iou_pairs`` maps decision *position* tuples ``(i, j)`` with ``i < j`` to
    their Stage 1 box IoU. Pairs with ``IoU >= params.panel_instance_iou`` are
    the same physical character counted twice (face+body of one detection
    chain) and are exempt from mutual suppression; every other pair is
    spatially distinct. When ``iou_pairs`` is None, all pairs count as
    distinct. ``allow_doppelganger=True`` disables the rule for
    doppelganger chapters. A demoted core becomes boundary and its reason
    records the outranking winner; the input list is not mutated.
    """
    if len(decisions) != len(panel_ids):
        raise ValueError(
            f"{len(decisions)} decisions but {len(panel_ids)} panel ids"
        )
    _require_decision_keys(decisions)
    for key, value in (iou_pairs or {}).items():
        key_valid = (
            isinstance(key, tuple) and len(key) == 2 and key[0] >= 0 and key[0] < key[1]
        )
        if not key_valid or not (0.0 <= float(value) <= 1.0):
            raise ValueError(
                f"iou_pairs must map ascending position tuples to IoU in [0, 1], "
                f"got {key}: {value}"
            )
        if key[1] >= len(decisions):
            raise ValueError(
                f"iou_pairs position {key} references decision {key[1]} but there "
                f"are only {len(decisions)} decisions"
            )

    # Pure outputs (Law 3): every decision dict and reasons list is copied.
    result = [
        {**decision, "reasons": list(decision.get("reasons", []))}
        for decision in decisions
    ]
    if allow_doppelganger:
        return result

    positions_by_panel: dict = {}
    for position, panel in enumerate(panel_ids):
        positions_by_panel.setdefault(panel, []).append(position)

    for panel, positions in positions_by_panel.items():
        cores = [p for p in positions if result[p]["tier"] == TIER_CORE]
        if len(cores) < 2:
            continue
        # Rank the panel winner by the PRE-boost "raw_score" when present:
        # a malicious-twin lookalike lifted past tau_core by the page prior
        # must never displace a natural core (plan rule 2 boosts
        # partial/back instances toward core, it does not outrank them).
        def _rank_score(position: int) -> float:
            decision = result[position]
            return float(decision.get("raw_score", decision["score"]))

        ranked = sorted(cores, key=lambda p: (-_rank_score(p), p))
        winner, kept = ranked[0], [ranked[0]]
        for candidate in ranked[1:]:
            overlaps_a_kept_core = any(
                iou_pairs is not None
                and float(iou_pairs.get((min(k, candidate), max(k, candidate)), 0.0))
                >= params.panel_instance_iou
                for k in kept
            )
            if overlaps_a_kept_core:
                # Same physical character as a kept core: keep both.
                kept.append(candidate)
                continue
            demoted = result[candidate]
            demoted["tier"] = TIER_BOUNDARY
            demoted["reasons"].append(
                f"same_panel_cannot_link: panel '{panel}' already has core "
                f"'{result[winner]['instance_id']}' (raw score "
                f"{_rank_score(winner):.4f} >= {_rank_score(candidate):.4f})"
            )
    return result


def apply_page_priors(decisions: list[dict], page_ids, boost: float) -> list[dict]:
    """Same-Page Prior Boost: +boost (capped at 1.0) on non-core scores.

    Boost applies only to instances on pages that contain at least one core
    decision, and only to instances that are not core themselves. Cores are
    never boosted (they already pass ``tau_core``), and a page's boost comes
    from the cores detected in the first-tier pass, not from instances that
    cross the threshold because of the boost itself.
    """
    boost = float(boost)
    if not np.isfinite(boost) or boost < 0.0:
        raise ValueError(f"page boost must be a finite value >= 0, got {boost}")
    if len(decisions) != len(page_ids):
        raise ValueError(f"{len(decisions)} decisions but {len(page_ids)} page ids")
    _require_decision_keys(decisions)

    core_pages = {
        page for decision, page in zip(decisions, page_ids)
        if decision["tier"] == TIER_CORE
    }
    result: list[dict] = []
    for decision, page in zip(decisions, page_ids):
        if page in core_pages and decision["tier"] != TIER_CORE and boost > 0.0:
            boosted_score = min(float(decision["score"]) + boost, 1.0)
            result.append(
                {
                    **decision,
                    "score": boosted_score,
                    "reasons": [
                        *decision.get("reasons", []),
                        f"page_prior_boost: +{boost:.3f} on page with a core "
                        f"(capped at 1.0)",
                    ],
                }
            )
        else:
            result.append({**decision, "reasons": list(decision.get("reasons", []))})
    return result


def _record_iou_of(box_a, box_b) -> float:
    """IoU of two [x1, y1, x2, y2] pixel instance boxes."""
    ax1, ay1, ax2, ay2 = (float(c) for c in box_a)
    bx1, by1, bx2, by2 = (float(c) for c in box_b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def _instance_iou_pairs(records: list[dict]) -> dict[tuple[int, int], float]:
    """IoU map for same-panel record pairs that both carry instance boxes.

    Records may hold an optional ``bbox`` (the instance's outermost Stage 1
    box, ``body`` > ``head`` > ``face``). Pairs missing either box stay out
    of the map: :func:`apply_panel_constraints` then counts them as
    spatially distinct, matching the documented no-box fallback.

    Known consequence of the body-precedence footprint: two heads branched
    inside one wide body share that body box (IoU 1.0) and are therefore
    exempt from mutual suppression — by plan rule 1's letter, a pair with
    overlapping boxes is not "spatially distinct", so the cannot-link
    constraint does not apply to it. Pairs are grouped by panel first so
    the scan stays O(sum(panel_size^2)) instead of O(records^2).
    """
    positions_by_panel: dict = {}
    for position, record in enumerate(records):
        if record.get("bbox") is None:
            continue
        positions_by_panel.setdefault(record["panel"], []).append(position)

    pairs: dict[tuple[int, int], float] = {}
    for positions in positions_by_panel.values():
        for index, i in enumerate(positions):
            box_a = records[i]["bbox"]
            for j in positions[index + 1:]:
                pairs[(i, j)] = _record_iou_of(box_a, records[j]["bbox"])
    return pairs


def assign_izutsumi(
    records: list[dict],
    bank: PrototypeBank,
    params: ClusteringParams,
    allow_doppelganger: bool = False,
) -> list[dict]:
    """End-to-end two-pass assignment: scored -> page priors -> tiers -> panels.

    Record contract (IN): dicts holding ``instance_id``, ``page``, ``panel``,
    ``embedding`` and ``crop_path``, plus the optional ``bbox``
    (``[x1, y1, x2, y2]`` pixel box of the instance's outermost part) used
    for Same-Panel spatial distinctness: same-panel pairs with box IoU >=
    ``params.panel_instance_iou`` are one physical character chain and are
    exempt from mutual suppression; pairs without both boxes count as
    distinct. Returned decisions carry ``instance_id``, ``score``, ``tier``,
    ``reasons``, ``subcluster`` and ``crop_path``. ``page`` / ``panel``
    default to the record's ``crop_path`` / ``page`` exactly as documented
    for the no-panel-segmentation flow.

    Call order (asserted by tests): raw :meth:`PrototypeBank.max_similarity`
    for every record -> first-tier split (:func:`assign_two_tier`) ->
    :func:`apply_page_priors` -> final :func:`assign_two_tier` ->
    :func:`apply_panel_constraints` with IoU pairs derived from record
    bboxes (demotes surplus cores; constraints win
    over priors) -> :func:`subcluster_embeddings` purely to annotate the
    ``subcluster`` field used by the review gallery.
    """
    _validate_tiers(params)
    if not records:
        return []
    for position, record in enumerate(records):
        missing = [key for key in _ASSIGN_RECORD_KEYS if key not in record]
        if missing:
            raise ValueError(
                f"record at position {position} is missing key(s) {missing}"
            )

    raw_scores = [bank.max_similarity(record["embedding"]) for record in records]
    pages = [record["page"] for record in records]
    panels = [record["panel"] for record in records]

    first_tier = assign_two_tier(
        [record["instance_id"] for record in records], raw_scores, params
    )
    boosted = apply_page_priors(first_tier, pages, params.page_prior_boost)

    final_tier = assign_two_tier(
        [record["instance_id"] for record in records],
        [decision["score"] for decision in boosted],
        params,
    )
    # Keep the page-prior reasons beneath the final-band rationale.
    for final, prior in zip(final_tier, boosted):
        final["reasons"] = [*prior.get("reasons", []), *final["reasons"]]
    # Pre-boost score: the panel cannot-link ranking uses this so a
    # lookalike lifted past tau_core by the page prior can never
    # displace a natural core (plan rule 2 semantics).
    for final, raw in zip(final_tier, raw_scores):
        final["raw_score"] = float(raw)

    constrained = apply_panel_constraints(
        final_tier, panels, params, iou_pairs=_instance_iou_pairs(records),
        allow_doppelganger=allow_doppelganger,
    )

    # The sub-clustering annotation only feeds the boundary review gallery
    # grouping — scores and tiers are already decided above. Degrade to
    # "Ungrouped" (noise label) instead of failing `catnip match` outright
    # when the instance count exceeds the dense-HDBSCAN budget.
    try:
        subclusters = subcluster_embeddings(
            np.stack([_as_embedding_vector(record["embedding"]) for record in records]),
            k_neighbors=params.k_neighbors,
            min_cluster_size=params.min_cluster_size,
        )
    except ValueError as err:
        if len(records) <= MAX_SUBCLUSTER_INSTANCES:
            raise
        logger.warning(
            "sub-cluster annotation skipped (%s); %d instances render as "
            "'Ungrouped' in the review gallery",
            err, len(records),
        )
        subclusters = np.full(len(records), -1, dtype=int)

    decisions: list[dict] = []
    for decision, record, subcluster in zip(constrained, records, subclusters):
        decisions.append(
            {
                **decision,
                "subcluster": int(subcluster),
                "crop_path": record["crop_path"],
            }
        )
    return decisions


# ---------------------------------------------------------------------------
# Boundary review gallery
# ---------------------------------------------------------------------------


def build_review_gallery(decisions: list[dict], crops: dict, out_html) -> Path:
    """Self-contained (no JS, inline CSS) HTML gallery of boundary decisions.

    Boundary instances are grouped into one section per HDBSCAN subcluster
    (noise/unassigned behind a trailing "Ungrouped" section); crop images
    are referenced with paths relative to ``out_html``. ``crops`` maps
    ``instance_id`` to the crop path; a boundary id missing from ``crops``
    renders a placeholder instead of failing the whole gallery. Returns the
    written file path.
    """
    boundary = [decision for decision in decisions if decision.get("tier") == TIER_BOUNDARY]

    by_subcluster: dict[int, list[dict]] = {}
    for decision in boundary:
        label = int(decision.get("subcluster", -1))
        by_subcluster.setdefault(label, []).append(decision)
    ungrouped = by_subcluster.pop(-1, [])
    sections = [
        (label, instances)
        for label, instances in sorted(by_subcluster.items())
        if instances
    ]
    if ungrouped:
        sections.append((-1, ungrouped))

    out_path = Path(out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not boundary:
        body = "<p class='empty'>No boundary instances to review.</p>"
    else:
        cards = []
        for label, instances in sections:
            title = (
                f"Subcluster {label}" if label != -1 else "Ungrouped (noise)"
            )
            cards.append(
                f"<h2>{html.escape(title)} "
                f"<span class='count'>{len(instances)} instance(s)</span></h2>"
            )
            gallery = ["<div class='gallery'>"]
            for decision in sorted(
                instances, key=lambda d: (-float(d.get("score", 0.0)), str(d["instance_id"]))
            ):
                gallery.append(_gallery_card(decision, crops, out_path))
            gallery.append("</div>")
            cards.append("".join(gallery))
        body = "\n".join(cards)

    out_path.write_text(_GALLERY_TEMPLATE.format(body=body), encoding="utf-8")
    logger.info("Review gallery: %d boundary instance(s) -> %s", len(boundary), out_path)
    return out_path


def _gallery_card(decision: dict, crops: dict, out_path: Path) -> str:
    instance_id = decision["instance_id"]
    score = float(decision.get("score", 0.0))
    crop_path = crops.get(instance_id)
    caption = html.escape(f"{instance_id} - {score:.4f}")
    if crop_path is None:
        return (
            f"<figure class='card missing'><div class='placeholder'>crop unavailable"
            f"</div><figcaption>{caption}</figcaption></figure>"
        )
    relative = os.path.relpath(str(crop_path), start=str(out_path.parent))
    src = html.escape(_url_quote(relative), quote=True)
    return (
        f"<figure class='card'><img loading='lazy' src='{src}' alt='{caption}'>"
        f"<figcaption>{caption}</figcaption></figure>"
    )


_GALLERY_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Catnip - Boundary Review Gallery</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 2rem auto 4rem; max-width: 1200px; color: #1c1e21; }}
  h1 {{ font-size: 1.4rem; }}
  h2 {{ font-size: 1.05rem; margin: 2.2rem 0 0.8rem; border-bottom: 1px solid #ddd;
        padding-bottom: 0.35rem; }}
  .count {{ color: #666; font-weight: normal; font-size: 0.85rem; }}
  .gallery {{ display: flex; flex-wrap: wrap; gap: 0.9rem; }}
  .card {{ margin: 0; width: 176px; background: #fff; border: 1px solid #ddd;
           border-radius: 6px; padding: 6px; }}
  .card img {{ width: 100%; height: 176px; object-fit: contain; background: #f5f5f5; }}
  .card.missing {{ outline: 1px dashed #b58900; }}
  .placeholder {{ height: 176px; display: flex; align-items: center; justify-content: center;
                  color: #999; font-size: 0.8rem; background: #fafafa; }}
  figcaption {{ font-size: 0.75rem; padding-top: 5px; word-break: break-all; }}
  .empty {{ color: #666; }}
</style>
</head>
<body>
<h1>Catnip - Boundary Review Gallery</h1>
<p>Boundary-tier instances grouped by HDBSCAN subcluster. Accept or reject an
entire subcluster (e.g. a new outfit or fanart style) in one action.</p>
{body}
</body>
</html>
"""
