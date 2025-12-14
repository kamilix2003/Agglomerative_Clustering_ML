
===========================
Agglomerative_Clustering_ML
===========================


    Agglomerative clustering algorithm implemenation in Python.


Project is part of Programming in Python Language course.

=============
Example usage
=============

.. code-block::

    from agglomerative_clustering_ml.agglomerative_clustering import agglomerative

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


.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.6. For details and usage
information on PyScaffold see https://pyscaffold.org/.
