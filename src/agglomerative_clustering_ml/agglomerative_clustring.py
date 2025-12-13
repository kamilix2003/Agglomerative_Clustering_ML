#!/usr/bin/env python3
# agglomerative.py
"""
Agglomerative clustering (NumPy-only) with multiple linkages and SciPy-style linkage output.

This module provides a straightforward, vectorized implementation of agglomerative
(hierarchical) clustering. It stores the full pairwise distance matrix (O(n^2) memory),
uses a heap (lazy deletion) to choose merges, and updates inter-cluster distances
via Lance–Williams-like updates (vectorized). Supported linkages:
    - 'single', 'complete', 'average'

Doxygen-style docstrings are used (with @param / @return tags).
"""

from typing import Tuple, List, Optional
import numpy as np
import heapq

__all__ = [
    "compute_pairwise_distances",
    "init_clusters",
    "pair_to_heap_entries",
    "lance_williams_update",
    "extract_min_pair",
    "merge_clusters",
    "agglomerative",
]

_supported_linkages = {"single", "complete", "average"}

def compute_pairwise_distances(X: np.ndarray) -> np.ndarray:
    """
    Compute full pairwise Euclidean distance matrix for rows of X.

    @param X: 2D array, shape (n_samples, n_features). Rows are observations.
    @return: 2D array D shape (n_samples, n_samples) where D[i, j] is the Euclidean
             distance between X[i] and X[j]. The diagonal entries are zero.
    """
    X = np.asarray(X, dtype=float)
    sq = np.sum(X * X, axis=1, keepdims=True)  # (n,1)
    D2 = sq + sq.T - 2.0 * (X @ X.T)
    # Numerical safety: clip small negatives to zero
    D2[D2 < 0] = 0.0
    D = np.sqrt(D2, dtype=float)
    return D


def init_clusters(n: int) -> Tuple[np.ndarray, np.ndarray, List[set], Optional[np.ndarray]]:
    """
    Initialize cluster bookkeeping structures.

    @param n: Number of initial clusters (typically = number of samples).

    @return: A tuple (active, sizes, members)
        - active: boolean array length n (True indicates cluster is active)
        - sizes: integer array length n (cluster sizes)
        - members: list of sets; members[i] contains original sample indices in cluster i
    """
    active = np.ones(n, dtype=bool)
    sizes = np.ones(n, dtype=int)
    members: List[set] = [{i} for i in range(n)]
    return active, sizes, members


def pair_to_heap_entries(D: np.ndarray) -> List[Tuple[float, int, int]]:
    """
    Create heap entries (distance, i, j) for the upper triangle of D (i < j),
    and heapify them for efficient pop-min.

    @param D: symmetric distance matrix (n, n) with floats.
    @return: list suitable for heapq operations (heapified).
    """
    n = D.shape[0]
    entries: List[Tuple[float, int, int]] = []
    for i in range(n):
        # push only upper-triangle pairs i < j
        for j in range(i + 1, n):
            entries.append((float(D[i, j]), i, j))
    heapq.heapify(entries)
    return entries


def lance_williams_update(linkage: str,
                          d_ik: float, d_jk: float, 
                          size_i: int, size_j: int) -> float:
    """
    Simple Lance–Williams update for the common linkages single/complete/average.

    @param linkage: 'single' | 'complete' | 'average'
    @param d_ik: distance between cluster i and k
    @param d_jk: distance between cluster j and k
    @param d_ij: distance between cluster i and j
    @param size_i: size of cluster i (int)
    @param size_j: size of cluster j (int)
    @param size_k: size of cluster k (int)
    @return: updated distance d(iuj, k)
    """
    if linkage == "single":
        return min(d_ik, d_jk)
    elif linkage == "complete":
        return max(d_ik, d_jk)
    elif linkage == "average":
        return (size_i * d_ik + size_j * d_jk) / (size_i + size_j)
    else:
        raise ValueError("Unsupported linkage for lance_williams_update: " + str(linkage))


def extract_min_pair(heap: List[Tuple[float, int, int]], D: np.ndarray, active: np.ndarray, tol: float = 1e-12) -> Tuple[int, int, float]:
    """
    Pop from heap until a valid active pair (i, j) with up-to-date distance is found.

    Lazy deletion: many heap entries can be stale after merges; this function skips
    them until it finds an active pair matching the current D[i, j].

    @param heap: heap list managed with heapq
    @param D: current inter-cluster distance matrix (n, n)
    @param active: boolean mask of active clusters
    @param tol: absolute tolerance for considering a popped distance equal to D[i,j]
    @return: tuple (i, j, distance)
    @raises RuntimeError: if heap is exhausted (should not happen normally)
    """
    n = D.shape[0]
    while heap:
        dist, i, j = heapq.heappop(heap)
        if i > j:
            i, j = j, i
        if not (0 <= i < n and 0 <= j < n):
            continue
        if not (active[i] and active[j]):
            continue
        current = D[i, j]
        if np.isfinite(current) and abs(dist - current) <= max(tol, 1e-12 * (1.0 + abs(current))):
            return int(i), int(j), float(current)
        # else stale -> skip
    raise RuntimeError("Heap exhausted without finding a valid pair.")


def merge_clusters(i: int, j: int,
                   active: np.ndarray,
                   sizes: np.ndarray,
                   D: np.ndarray,
                   linkage: str,
                   heap: List[Tuple[float, int, int]],
                   members: Optional[List[set]] = None) -> None:
    """
    Merge cluster j into cluster i. Update active mask, sizes, D, members,
    and push updated heap entries (i, k) for all remaining active k.

    @param i: index of cluster to keep (int)
    @param j: index of cluster to deactivate (int). j != i.
    @param active: boolean mask of active clusters; modified in-place
    @param sizes: integer array of cluster sizes; modified in-place
    @param D: inter-cluster distance matrix (n, n); modified in-place
    @param linkage: linkage method string
    @param heap: heap list to push updated pairs into (heapq used); modified in-place
    @param members: optional list of sets for members; updated in-place if provided
    @return: None
    """
    if i == j:
        raise ValueError("Cannot merge a cluster with itself.")
    if not (active[i] and active[j]):
        raise ValueError("Both clusters must be active to merge.")

    size_i = int(sizes[i])
    size_j = int(sizes[j])
    size_new = size_i + size_j

    act_idx = np.where(active)[0]
    others = act_idx[(act_idx != i) & (act_idx != j)]

    # Vectorized update depending on linkage
    d_ik = D[i, others]
    d_jk = D[j, others]
    
    if linkage in _supported_linkages:
        d_new = lance_williams_update(linkage, d_ik, d_jk, size_i, size_j)
    else:
        raise ValueError(f"Unsupported linkage: {linkage}")

    # write back for i <-> others
    D[i, others] = d_new
    D[others, i] = d_new

    # deactivate j: set its distances to +inf and update active/sizes
    D[j, :] = np.inf
    D[:, j] = np.inf
    active[j] = False
    sizes[i] = size_new
    sizes[j] = 0

    # update members if provided
    if members is not None:
        members[i] = members[i].union(members[j])
        members[j] = set()

    # push updated heap entries for pairs (i, k)
    for k in others:
        a = min(i, int(k))
        b = max(i, int(k))
        heapq.heappush(heap, (float(D[a, b]), a, b))


def agglomerative(X: np.ndarray,
                  n_clusters: int = 1,
                  linkage: str = "average",
                  return_linkage: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Perform agglomerative clustering on data matrix X.

    @param X: data matrix shape (n_samples, n_features)
    @param n_clusters: desired number of clusters (1 <= n_clusters <= n_samples)
    @param linkage: one of 'single', 'complete', 'average'
    @param return_linkage: if True, also return SciPy-style linkage matrix Z shape (n-1, 4)
                           with rows [idx1, idx2, dist, new_cluster_size]

    @return: tuple (labels, linkage_matrix_or_None)
        - labels: integer array shape (n_samples,) with labels 0..(n_clusters-1)
        - linkage_matrix_or_None: np.ndarray shape (n-1, 4) if return_linkage else None
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array (n_samples, n_features).")
    n, d = X.shape
    if not (1 <= n_clusters <= n):
        raise ValueError("n_clusters must be between 1 and n_samples.")

    # initial pairwise distances, diagonal set to +inf so they aren't picked
    D = compute_pairwise_distances(X)
    np.fill_diagonal(D, np.inf)

    active, sizes, members = init_clusters(n)
    heap = pair_to_heap_entries(D)

    # bookkeeping for linkage matrix (if requested)
    if return_linkage:
        node_id = list(range(n))  # current mapping: cluster index -> node id
        Z_rows: List[List[float]] = []
        next_node = n
    else:
        node_id = None
        Z_rows = None
        next_node = n  # unused

    num_active = n
    while num_active > n_clusters:
        i, j, dist = extract_min_pair(heap, D, active)
        # record linkage row (SciPy-style) if desired
        if return_linkage:
            a = node_id[i]
            b = node_id[j]
            new_size = int(sizes[i] + sizes[j])
            Z_rows.append([float(a), float(b), float(dist), float(new_size)])
            # the surviving cluster i gets assigned the new node id
            node_id[i] = next_node
            next_node += 1

        # merge j into i
        merge_clusters(i, j, active, sizes, D, linkage, heap, members)
        if return_linkage:
            node_id[j] = -1  # mark j as inactive in the node mapping

        num_active -= 1

    # produce contiguous labels for the active clusters
    labels = np.empty(n, dtype=int)
    active_indices = [idx for idx in range(n) if active[idx]]
    for label, idx in enumerate(active_indices):
        for mem in members[idx]:
            labels[mem] = label

    if return_linkage:
        Z = np.array(Z_rows, dtype=float)
        return labels, Z
    else:
        return labels, None
