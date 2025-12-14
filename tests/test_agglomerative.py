import numpy as np
import pytest
from agglomerative_clustering_ml.agglomerative_clustering import (
    compute_pairwise_distances,
    init_clusters,
    pair_to_heap_entries,
    lance_williams_update,
    extract_min_pair,
    merge_clusters,
    agglomerative
)

def test_pairwise_distances_basic_properties():
    """
    Verify basic mathematical properties of pairwise distances.

    Checks:
    - correct output shape
    - zeros on the diagonal
    - symmetry
    - correctness on a known example
    """
    X = np.array([[0, 0], [3, 4], [6, 8]])
    D = compute_pairwise_distances(X)
    
    assert D.shape == (3, 3)

    assert np.allclose(np.diag(D), 0)

    assert np.allclose(D, D.T)

    assert pytest.approx(D[0, 1]) == 5.0
    assert pytest.approx(D[1, 0]) == 5.0

def test_pairwise_distances_matches_naive():
    """
    Compare vectorized distance computation with a naive reference
    implementation using explicit loops.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(7, 3))
    D = compute_pairwise_distances(X)

    Dn = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            Dn[i, j] = np.linalg.norm(X[i] - X[j])

    assert np.allclose(D, Dn, atol=1e-6)


def test_init_clusters():
    """
    Test correct initialization of clusters.

    Verifies:
    - all clusters start active
    - each cluster has size 1
    - each cluster initially contains exactly one member
    """

    active, sizes, members = init_clusters(4)
    assert active.dtype == bool
    assert active.shape == (4,)
    assert np.all(active)

    assert sizes.shape == (4,)
    assert np.all(sizes == 1)

    assert isinstance(members, list)
    assert len(members) == 4
    assert members[0] == {0}
    assert members[3] == {3}

def test_pair_to_heap_entries_size_and_min():
    """
    Verify heap construction from a distance matrix.

    Checks:
    - correct number of pair entries
    - smallest distance pair is at the top of the heap
    """
    X = np.array([[0.0, 0.0],
                  [1.0, 0.0],
                  [10.0, 0.0]])
    D = compute_pairwise_distances(X)
    heap = pair_to_heap_entries(D)

    n = D.shape[0]
    assert len(heap) == n * (n - 1) // 2

    dist, i, j = heap[0]
    assert pytest.approx(dist, abs=1e-12) == 1.0
    assert {i, j} == {0, 1}


@pytest.mark.parametrize(
    "linkage, expected",
    [
        ("single", 2.0),
        ("complete", 5.0),
        ("average", (1 * 2.0 + 3 * 5.0) / (1 + 3)),
    ],
)
def test_lance_williams_update_scalar(linkage, expected):
    """
    Test Lance-Williams distance update formula for supported linkage types.
    """
    d_ik = 2.0
    d_jk = 5.0
    size_i = 1
    size_j = 3
    out = lance_williams_update(linkage, d_ik, d_jk, size_i, size_j)
    assert pytest.approx(out, abs=1e-12) == expected

def test_lance_williams_update_invalid_linkage():
    with pytest.raises(ValueError):
        lance_williams_update("weird", 1.0, 2.0, 1, 1)

def test_extract_min_pair_skips_stale_entries():
    """
    Ensure extract_min_pair ignores stale heap entries.

    After modifying the distance matrix, outdated heap entries should
    be skipped until a valid active pair is found.
    """
    X = np.array([[0.0, 0.0],
                  [1.0, 0.0],
                  [5.0, 0.0]])
    D = compute_pairwise_distances(X)
    active, sizes, members = init_clusters(3)
    heap = pair_to_heap_entries(D)

    D[0, 1] = 100.0
    D[1, 0] = 100.0

    i, j, dist = extract_min_pair(heap, D, active)

    assert {i, j} in ({1, 2}, {0, 2})
    assert pytest.approx(dist, abs=1e-12) == D[i, j]


def test_merge_clusters_updates_state_average_linkage():
    """
    Test correct state updates after merging clusters using average linkage.
    """
    X = np.array([[0.0, 0.0],
                  [1.0, 0.0],
                  [10.0, 0.0]])
    D = compute_pairwise_distances(X)
    active, sizes, members = init_clusters(3)
    heap = pair_to_heap_entries(D)

    i, j, dist = extract_min_pair(heap, D, active)
    assert {i, j} == {0, 1}

    merge_clusters(i, j, active, sizes, D, linkage="average", heap=heap, members=members)

    assert active[i] 
    assert not active[j]
    assert sizes[i] == 2
    assert sizes[j] == 0

    other = 2
    expected = (1 * compute_pairwise_distances(X)[i, other] + 1 * compute_pairwise_distances(X)[j, other]) / 2.0
    assert pytest.approx(D[i, other], rel=1e-10, abs=1e-10) == expected
    assert pytest.approx(D[other, i], rel=1e-10, abs=1e-10) == expected

def test_merge_clusters_rejects_invalid():
    X = np.array([[0.0, 0.0],
                  [1.0, 0.0]])
    D = compute_pairwise_distances(X)
    active, sizes, members = init_clusters(2)
    heap = pair_to_heap_entries(D)

    with pytest.raises(ValueError):
        merge_clusters(0, 0, active, sizes, D, linkage="average", heap=heap, members=members)

    with pytest.raises(ValueError):
        merge_clusters(0, 1, active=np.array([True, False]), sizes=sizes, D=D, linkage="average", heap=heap, members=members)

    with pytest.raises(ValueError):
        merge_clusters(0, 1, active, sizes, D, linkage="unknown", heap=heap, members=members)

def test_agglomerative_returns_labels_and_optional_linkage():
    """
    End-to-end test of agglomerative clustering.

    Verifies:
    - correct label shape and type
    - correct linkage matrix shape when return_linkage=True
    """
    rng = np.random.default_rng(0)
    A = rng.normal(loc=0.0, scale=0.2, size=(10, 2))
    B = rng.normal(loc=3.0, scale=0.2, size=(10, 2))
    X = np.vstack([A, B])

    n_clusters = 2

    labels, Z = agglomerative(X, n_clusters=n_clusters, linkage="average", return_linkage=True)

    assert labels.shape == (X.shape[0],)
    assert labels.dtype == int

    assert isinstance(Z, np.ndarray)
    assert Z.shape == (X.shape[0] - n_clusters, 4)

def test_agglomerative_invalid_n_clusters():
    X = np.zeros((5, 2))
    with pytest.raises(ValueError):
        agglomerative(X, n_clusters=0)
    with pytest.raises(ValueError):
        agglomerative(X, n_clusters=6)