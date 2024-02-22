import numpy as np
from numpy.linalg import norm
from scipy.sparse import issparse, csr_matrix


def cosine_distance(v1, v2):
    """Compute the cosine distance between the two input vectors.

    Args:
        v1: A (sparse or dense) vector.
        v2: Another (sparse or dense) vector.

    Returns:
        float: The cosine distance between `v1` and `v2`.
    """
    # TODO: compute the cosine distance between two input vectors
    #       the implementation should work for both sparse and
    #       dense input vectors.

    # Check if the vectors are sparse and if so, convert them to dense
    if issparse(v1):
        v1 = v1.toarray()
    if issparse(v2):
        v2 = v2.toarray()

    # Compute the inner product of the two vectors
    inner_product = np.dot(v1, v2.T)

    # Compute the magnitudes (or norms) of the vectors
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # Compute cosine similarity
    cosine_similarity = inner_product / (norm_v1 * norm_v2)

    # Calculate cosine distance
    dist = 1 - cosine_similarity

    return dist

    # # Check if the vectors are sparse and if so, convert them to dense
    # try:
    #     if issparse(v1):
    #         v1 = v1.toarray().ravel()
    #     if issparse(v2):
    #         v2 = v2.toarray().ravel()
    # except ImportError:
    #     pass  # If scipy is not imported, assume the vectors are not sparse
    #
    # # Compute the inner product of the two vectors
    # inner_product = np.dot(v1, v2)
    #
    # # Compute the magnitudes (or norms) of the vectors
    # magnitude_v1 = norm(v1)
    # magnitude_v2 = norm(v2)
    #
    # # Compute cosine similarity
    # cosine_similarity = inner_product / (magnitude_v1 * magnitude_v2)
    #
    # # Calculate cosine distance
    # dist = 1 - cosine_similarity

    # # If v2 is 0 vector
    # if np.all(v2 == 0):
    #     return 1
    #
    # # Initialize inner product, product of Euclidean for v1 and v2
    # inner_product = 0
    # v1_product_Eu = 0
    # v2_product_Eu = 0
    #
    # # print(type(v1))
    # # print(type(v2))
    # # print(v2)
    #
    # # loop each part of v1 and v2 for inner product and Euclidean
    # for col1, value1 in zip(v1.indices, v1.data):
    #     for i in range(len(v2)):
    #         if col1 == i:
    #             # Calculate part of inner product and add them all
    #             inner_product += value1 * v2[i]
    #
    #             # Calculate prat of product of Euclidean for v1 and v2 and add them together
    #             v1_product_Eu += value1 ** 2
    #             v2_product_Eu += v2[i] ** 2
    #
    # # Calculate Euclidean for each v1 and v2
    # v1_Eu = v1_product_Eu ** 0.5
    # v2_Eu = v2_product_Eu ** 0.5
    #
    # # Calculate dist
    # dist = 1 - inner_product / (v1_Eu * v2_Eu)
    #
    # return dist

    # # If v2 is a zero vector
    # if np.all(v2 == 0):
    #     return 1.0
    #
    # # Initialize inner product, norm of v1, and norm of v2
    # inner_product = 0
    # v1_norm = 0
    # v2_norm = 0
    #
    # if isinstance(v1, csr_matrix) and isinstance(v2, np.ndarray):
    #
    #     for index, value in zip(v1.indices, v1.data):
    #         inner_product += value * v2[index]
    #         v1_norm += value ** 2
    #         v2_norm += v2[index] ** 2
    # else:
    #     for index in range(len(v1)):
    #         inner_product += v1[index] * v2[index]
    #         v1_norm += v1[index]**2
    #         v2_norm += v2[index]**2
    #
    #
    # # Finish computing the norms
    # v1_norm = v1_norm ** 0.5
    # v2_norm = v2_norm ** 0.5
    #
    # # Avoid division by zero
    # if v1_norm == 0 or v2_norm == 0:
    #     return 1.0
    #
    # # Calculate cosine similarity
    # cos_sim = inner_product / (v1_norm * v2_norm)
    #
    # # Calculate cosine distance
    # dist = 1.0 - cos_sim
    # return dist


def compute_distances(data, centroids):
    """compute the cosine distances between every data point and
    every centroid.

    Args:
        data: A (sparse or dense) matrix of features for N documents.
            Each row represents a document.
        centroids (np.ndarray): The K cluster centres. Each row
            represent a cluster centre.

    Returns:
        np.ndarray: An N x K matrix of cosine distances.
    """
    # check the input
    assert data.shape[1] == centroids.shape[1]

    N = data.shape[0]
    K = centroids.shape[0]
    dists = np.full((N, K), -1.)

    # TODO: Compute the distances between data points and centroids
    #       such that dists[i, j] is the cosine distance between 
    #       the i-th data point and the j-th centroid.

    # Compute the distance between data points and centroids
    for i in range(N):
        for j in range(K):
            dists[i, j] = cosine_distance(data[i], centroids[j])

    return dists


def assign_data_points(distances):
    """Assign each data point to its closest centroid.

    Args:
        distances (np.ndarray): An N x K matrix where distances[i, j]
            is the cosine distance between the i-th data point and
            the j-th centroid.

    Returns:
        np.ndarray: A vector of size N.
    """
    N, K = distances.shape
    clusters = np.full(N, -1)

    # TODO: Assign each data point to its closest centroid such that
    #       clusters[i] = j denotes that the i-th data point is
    #       assigned to the j-th centroid.

    # Assign data point to its closest centroid
    for i in range(N):
        clusters[i] = np.argmin(distances[i])  # argmin return index of the min value of row vector

    return clusters


def update_centroids(data, centroids, clusters):
    """Re-compute each centroid as the average of the data points
    assigned to it.

    Args:
        data: A (sparse or dense) matrix of features for N documents.
            Each row represents a document.
        centroids (np.ndarray): The K cluster centres. Each row
            represent a cluster centre.
        clusters (np.ndarray): A vector of size N where clusters[i] = j
            denotes that the i-th data point is assigned to the j-th
            centroid.

    Returns:
        np.ndarray: The updated centroids.
    """
    # check the input
    assert data.shape[1] == centroids.shape[1]
    N = data.shape[0]
    K = centroids.shape[0]
    assert clusters.shape[0] == N

    # TODO: Re-compute each centroid as the average of the data points
    #       assigned to it.

    # Initialize new centroids
    new_centroids = np.zeros_like(centroids)

    for j in range(K):
        # Find all data points that belongs to j
        j_data = data[clusters == j]

        # Check if j_data is sparse
        if issparse(j_data):
            num_rows = j_data.shape[0]
        else:
            num_rows = len(j_data)

        # Re-compute the centroid
        if num_rows > 0:
            new_centroids[j] = np.mean(j_data, axis=0)
        else:
            new_centroids[j] = centroids[j]  # No data points belong to the centroid

    return new_centroids


def kmeans(data, K, max_iter=10, rng=None):
    """Clustering data points using the KMeans algorithm.

    Args:
        data: A matrix of features of documents. Each row represents a document.
        K (int): The number of cluster centres.
        max_iter (int): The maximum number of iterations to run in the KMeans algorithm.
        rng (np.random.Generator): A random number generator.

    Returns:
        centroids (np.ndarray): The cluster centres (after the re-computation of centroids).
        clusters (np.ndarray): The index of cluster each document belongs to, e.g., clusters[i] = k
            denotes that the i-th document is in the k-th cluster.
    """
    print(f'Clustering using KMeans (K={K}) ...')
    N = data.shape[0]
    assert N >= K
    rng = np.random.default_rng(rng)
    indices = rng.choice(N, size=K, replace=False)
    if issparse(data):
        centroids = data[indices, :].A  # dense
    else:
        centroids = data[indices, :]

    print(f'{"Iteration":>10} {"Total Distance":>20}')
    prev_clusters = None
    for i in range(max_iter):
        dists = compute_distances(data, centroids)
        clusters = assign_data_points(dists)
        centroids = update_centroids(data, centroids, clusters)
        print(f'{i:>10} {round(dists.min(axis=1).sum(), 2):>20}')
        if prev_clusters is not None and np.all(prev_clusters == clusters):
            return centroids, clusters
        prev_clusters = clusters
    return centroids, clusters
