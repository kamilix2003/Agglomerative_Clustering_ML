from agglomerative_clustering_ml.agglomerative_clustering import agglomerative

if __name__ == "__main__":
    # Example dataset
    X = [
        [1.0, 0.0],
        [9.0, 1.0],
        [1.0, 1.0],
        [6.0, 2.0],
        [5.0, 6.0],
    ]

    # Perform agglomerative clustering
    clusters, _ = agglomerative(X, n_clusters=3, linkage="average")

    for data, cluster in zip(X, clusters):
        print(f"Data point: {data}, Cluster: {cluster}")