import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

def plot_clusters(axis: Axes, X: np.ndarray, labels: np.ndarray) -> None:
    """
    Plots the clustered data points in 2D.

    Args:
        X (np.ndarray): Data points of shape (n_samples, 2).
        labels (np.ndarray): Cluster labels of shape (n_samples,).
    """
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_points = X[labels == label]
        axis.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}')
    
    plt.title('Agglomerative Clustering Results')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_dendrogram(Z: np.ndarray) -> None:
    """
    Plots the dendrogram for the hierarchical clustering.

    Args:
        Z (np.ndarray): Linkage matrix of shape (n_samples - 1, 4).
    """
    from scipy.cluster.hierarchy import dendrogram

    plt.figure(figsize=(10, 7))
    dendrogram(Z)
    plt.title('Dendrogram for Agglomerative Clustering')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.show()
    
if __name__ == "__main__":
    # Example usage
    from agglomerative_clustering_ml.agglomerative_clustring import agglomerative
    rng = np.random.RandomState(0)
    A = rng.normal(loc=0.0, scale=0.3, size=(10, 2))
    B = rng.normal(loc=2.0, scale=0.3, size=(8, 2))
    X = np.vstack([A, B])

    labels, Z = agglomerative(X, n_clusters=2, linkage='average', return_linkage=True)
    
    fig, ax = plt.subplots()
    plot_clusters(ax, X, labels)
    plt.show()
    # plot_dendrogram(Z)