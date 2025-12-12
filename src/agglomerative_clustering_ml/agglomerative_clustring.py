import numpy as np

from typing import *

from heapq import *

_supported_linkages = {'single', 'complete', 'average'}

def pairwise_distances(X: np.ndarray) -> np.ndarray:
  """ Return a (n,n) matrix of Euclidean distances for rows of X.

         | p1 | p2 | p3 | ... 
      -------------------
      p1 |0   |d12 |... |...
      p2 |d21 |0   |... |...
      p3 |d31 |d32 |0   |...
      
  Args:
      X (np.ndarray): pairs of points (n_samples, n_features)

  Returns:
      np.ndarray: pairwise distance matrix (n_samples, n_samples)
  """
  sq = np.sum(X**2, axis=1, keepdims=True)
  dist = sq + sq.T - 2 * np.dot(X, X.T)
  
  dist[dist < 0] = 0  # Numerical stability
  
  return np.sqrt(dist)

def init_clusters(n: int) -> Tuple[np.ndarray, np.ndarray, List[set]]:
  """ Initialize clusters for agglomerative clustering.

  Args:
      n (int): n clusters

  Returns:
      Tuple[np.ndarray, np.ndarray, List[set]]:   
      active: boolean mask of length n (True for active clusters)
      sizes: array of cluster sizes, shape (n,)
      members: list of sets, members[i] = {indices in cluster i}
  """
  active = np.ones(n, dtype=bool)
  sizes = np.ones(n, dtype=int)
  members = [{i} for i in range(n)]
  return active, sizes, members

def pair_to_heap_entries(D: np.ndarray) -> List[Tuple[float, int, int]]:
  """_summary_

  Args:
      D (np.ndarray): (n, n) pairwise distance matrix

  Returns:
      List[Tuple[float, int, int]]: list of (distance, i, j) tuples
  """
  
  n = D.shape[0]
  out = []
  
  for i in range(n):
    for j in range(i + 1, n):
      out.append((D[i, j], i, j))
      
  heapify(out)
  return out
  
def lance_williams_update(linkage: str,
                          d_ik: float, d_jk: float, d_ij: float,
                          size_i: int, size_j: int, size_k: int) -> float:
  """Computes new distance between merged cluster (i,j) and cluster k.

  Args:
      linkage (str): linkage type: 'single', 'complete', 'average'
      d_ik (float): distance between cluster i and k
      d_jk (float): distance between cluster j and k
      d_ij (float): distance between cluster i and j
      size_i (int): size of cluster i
      size_j (int): size of cluster j
      size_k (int): size of cluster k

  Returns:
      float: distance between merged cluster (i,j) and cluster k
  """
  
  if linkage == 'single':
    return np.minimum(d_ik, d_jk)
  elif linkage == 'complete':
    return np.maximum(d_ik, d_jk)
  elif linkage == 'average':
    return (size_i * d_ik + size_j * d_jk) / (size_i + size_j)
  else:
    raise ValueError(f"Unknown linkage type: {linkage}")
  
def extract_min_pair(heap,
                     D: np.ndarray,
                     active: np.ndarray,
                     tol: float = 1e-12) -> Tuple[int, int, float]:
  """_summary_

  Args:
      heap (_type_): _description_
      D (np.ndarray): _description_
      active (np.ndarray): _description_
      tol (float, optional): _description_. Defaults to 1e-12.

  Returns:
      Tuple[int, int, float]: _description_
  """
  n = D.shape[0]
  while heap:
      dist, i, j = heappop(heap)
      # ensure i < j
      if i > j:
          i, j = j, i
      # check valid indices
      if not (0 <= i < n and 0 <= j < n):
          continue
      if not (active[i] and active[j]):
          continue
      # check not stale (compare to current D)
      current = D[i, j]
      if np.isfinite(current) and abs(dist - current) <= max(tol, 1e-12 * (1.0 + abs(current))):
          return i, j, float(current)
      # else stale entry -> discard and continue
  raise RuntimeError("Heap exhausted without finding valid pair — something went wrong.")
  
def merge_clusters(i: int, j: int,
                   active: np.ndarray, sizes: np.ndarray,
                   D: np.ndarray, # current inter-cluster distances
                   linkage: str,
                   heap,
                   members: Optional[List[set]]=None) -> None:
  """_summary_

  Args:
      i (int): cluster index i
      j (int): cluster index j
      active (np.ndarray): active clusters mask
      sizes (np.ndarray): cluster sizes
      D (np.ndarray): current inter-cluster distances
      heap (_type_): heap of inter-cluster distances
      members (Optional[List[set]], optional): cluster members tracking. Defaults to None.
  """
  
  if i == j:
        raise ValueError("Cannot merge cluster with itself")
  if not (active[i] and active[j]):
      raise ValueError("Both clusters must be active to merge")

  # choose canonical ordering: keep i
  # compute new size
  size_i = sizes[i]
  size_j = sizes[j]
  size_new = size_i + size_j
  # vector of active indices excluding i and j
  act_idx = np.where(active)[0]
  # exclude i and j
  others = act_idx[(act_idx != i) & (act_idx != j)]
  
  if linkage not in _supported_linkages:
      raise ValueError(f"Unknown linkage type: {linkage}")
    
  d_new = lance_williams_update(linkage, 
                                D[i, others], D[j, others], D[i, j],
                                size_i, size_j, sizes[others])
  
  # write back updated distances for i <-> others
  D[i, others] = d_new
  D[others, i] = d_new

  # mark j inactive
  active[j] = False

  # update size
  sizes[i] = size_new
  sizes[j] = 0
  
  for k_idx, k in enumerate(others):
    a = min(i, int(k))
    b = max(i, int(k))
    heappush(heap, (float(D[a, b]), a, b))
  
def agglomerative(X: np.ndarray,
                  n_clusters: int = 1,
                  linkage: str = 'average',
                  return_linkage: bool = False) -> np.ndarray|Tuple[np.ndarray, np.ndarray]:
  """
  Returns labels (n,) or (labels, linkage_matrix) if requested.
  linkage_matrix shape (n-1, 4) same as scipy: [idx1, idx2, dist, size_new]
  """

  X = np.asarray(X, dtype=float)
  n, d = X.shape
  if not (1 <= n_clusters <= n):
      raise ValueError("n_clusters must be between 1 and n")

  # initial distance matrix between points
  D = pairwise_distances(X)

  active, sizes, members = init_clusters(n)
  heap = pair_to_heap_entries(D)

  # for linkage matrix bookkeeping
  if return_linkage:
      # We'll map cluster indices to "node ids" like SciPy:
      # initial node ids 0..n-1, new nodes n, n+1, ...
      node_id = np.arange(n, dtype=int).tolist()  # node_id[i] gives current node id for cluster index i
      Z_rows = []
      next_node = n
  else:
      node_id = None
      Z_rows = None
      next_node = n  # unused if not returning linkage

  num_active = n
  # main loop
  while num_active > n_clusters:
      i, j, dist = extract_min_pair(heap, D, active)
      # record linkage row if requested
      if return_linkage:
          a = node_id[i]
          b = node_id[j]
          new_size = int(sizes[i] + sizes[j])
          Z_rows.append([a, b, float(dist), new_size])
          # assign new node id to the surviving cluster i
          node_id[i] = next_node
          next_node += 1

      # merge j into i
      merge_clusters(i, j, active, sizes, D, linkage, heap, members)
      if return_linkage:
          # after merge, ensure node_id[j] is not used (cluster j inactive), but keep list size consistent
          node_id[j] = -1

      num_active -= 1

  # build labels from members of active clusters
  labels = np.empty(n, dtype=int)
  # map each active cluster to a label 0..m-1
  active_indices = [idx for idx in range(n) if active[idx]]
  for label, idx in enumerate(active_indices):
      for mem in members[idx]:
          labels[mem] = label

  if return_linkage:
      Z = np.array(Z_rows, dtype=float)
      return labels, Z
  else:
      return labels, None
  
if __name__ == "__main__":
    rng = np.random.RandomState(0)
    A = rng.normal(loc=0.0, scale=0.3, size=(10, 2))
    B = rng.normal(loc=2.0, scale=0.3, size=(8, 2))
    X = np.vstack([A, B])

    print("Testing average linkage, 2 clusters")
    labels, Z = agglomerative(X, n_clusters=2, linkage='average', return_linkage=True)
    print("labels:", labels)
    print("linkage (last 3 rows):\n", Z[-3:])


    