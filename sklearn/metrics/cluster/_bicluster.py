import numpy as np
from scipy import sparse as sp
from scipy.special import comb
from scipy.optimize import linear_sum_assignment

from ...utils.validation import check_consistent_length, check_array
from ._supervised import check_clusterings, contingency_matrix


__all__ = ["consensus_score", "coclustering_adjusted_rand_score"]


def _check_rows_and_columns(a, b):
    """Unpacks the row and column arrays and checks their shape."""
    check_consistent_length(*a)
    check_consistent_length(*b)
    checks = lambda x: check_array(x, ensure_2d=False)
    a_rows, a_cols = map(checks, a)
    b_rows, b_cols = map(checks, b)
    return a_rows, a_cols, b_rows, b_cols


def _jaccard(a_rows, a_cols, b_rows, b_cols):
    """Jaccard coefficient on the elements of the two biclusters."""
    intersection = (a_rows * b_rows).sum() * (a_cols * b_cols).sum()

    a_size = a_rows.sum() * a_cols.sum()
    b_size = b_rows.sum() * b_cols.sum()

    return intersection / (a_size + b_size - intersection)


def _pairwise_similarity(a, b, similarity):
    """Computes pairwise similarity matrix.

    result[i, j] is the Jaccard coefficient of a's bicluster i and b's
    bicluster j.

    """
    a_rows, a_cols, b_rows, b_cols = _check_rows_and_columns(a, b)
    n_a = a_rows.shape[0]
    n_b = b_rows.shape[0]
    result = np.array(
        list(
            list(
                similarity(a_rows[i], a_cols[i], b_rows[j], b_cols[j])
                for j in range(n_b)
            )
            for i in range(n_a)
        )
    )
    return result


def consensus_score(a, b, similarity="jaccard"):
    """The similarity of two sets of biclusters.

    Similarity between individual biclusters is computed. Then the
    best matching between sets is found using the Hungarian algorithm.
    The final score is the sum of similarities divided by the size of
    the larger set.

    Read more in the :ref:`User Guide <biclustering>`.

    Parameters
    ----------
    a : (rows, columns)
        Tuple of row and column indicators for a set of biclusters.

    b : (rows, columns)
        Another set of biclusters like ``a``.

    similarity : string or function, optional, default: "jaccard"
        May be the string "jaccard" to use the Jaccard coefficient, or
        any function that takes four arguments, each of which is a 1d
        indicator vector: (a_rows, a_columns, b_rows, b_columns).

    References
    ----------

    * Hochreiter, Bodenhofer, et. al., 2010. `FABIA: factor analysis
      for bicluster acquisition
      <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2881408/>`__.

    """
    if similarity == "jaccard":
        similarity = _jaccard
    matrix = _pairwise_similarity(a, b, similarity)
    row_indices, col_indices = linear_sum_assignment(1.0 - matrix)
    n_a = len(a[0])
    n_b = len(b[0])
    return matrix[row_indices, col_indices].sum() / max(n_a, n_b)


def coclustering_adjusted_rand_score(
    labels_true_part_1,
    labels_true_part_2,
    labels_pred_part_1,
    labels_pred_part_2,
):
    """Coclustering Adjusted Rand Index for two sets of biclusters.

    The Coclustering Adjuster Rand Index (CARI) computes a similarity measure
    between two coclusterings and is an adaptation of the
    Adjusted Rand Index (ARI) developed by Hubert and Arabie (1985) from a
    coclustering point of view.
    Like the ARI, this index is symmetric and takes the value 1 when the
    couples of partitions agree perfectly up to a permutation.

    Parameters
    ----------
    labels_true_part_1 : int array, shape = (n_samples_1,)
        Ground truth class labels of the first partition used as reference

    labels_true_part_2 : int array, shape = (n_samples_2,)
        Ground truth class labels of the second partition used as reference

    labels_pred_part_1 : int array, shape = (n_samples_1,)
        Cluster labels of the fist partition to evaluate

    labels_pred_part_2 : int array, shape = (n_samples_2,)
        Cluster labels of the second partition to evaluate

    Returns
    -------
    cari : float
       Similarity score between -1.0 and 1.0. Random labelings have a CARI
       close to 0.0. 1.0 stands for perfect match.

    Examples
    --------
    Perfectly matching labelings have a score of 1 even
      >>> from sklearn.metrics.cluster import coclustering_adjusted_rand_score
      >>> coclustering_adjusted_rand_score(
            [0, 0, 1, 1],
            [0, 0, 1, 1, 2, 2],
            [0, 0, 1, 1],
            [0, 0, 1, 1, 2, 2]
        )
      1.0
      >>> coclustering_adjusted_rand_score(
            [0, 0, 1, 1],
            [0, 0, 1, 1, 2, 2],
            [0, 0, 1, 1],
            [2, 2, 1, 1, 0, 0]
        )
      1.0
    If classes members are completely split across different clusters, the
    assignment is totally incomplete, hence the ARI is very low::
      >>> coclustering_adjusted_rand_score(
            [0, 0, 0, 0],
            [0, 0, 0],
            [0, 1, 2, 3],
            [2, 1, 0]
        )
      0.0

    References
    ----------
    .. [Robert2019] Valérie Robert, Yann Vasseur, Vincent Brault.
      Comparing high dimensional partitions with the Coclustering Adjusted Rand
      Index. 2019. https://hal.inria.fr/hal-01524832v4

    .. [Hubert1985] L. Hubert and P. Arabie, Comparing Partitions,
      Journal of Classification 1985
      https://link.springer.com/article/10.1007%2FBF01908075

    """

    labels_true_part_1, labels_pred_part_1 = check_clusterings(
        labels_true_part_1, labels_pred_part_1
    )
    labels_true_part_2, labels_pred_part_2 = check_clusterings(
        labels_true_part_2, labels_pred_part_2
    )
    n_samples_part_1 = labels_true_part_1.shape[0]
    n_samples_part_2 = labels_true_part_2.shape[0]

    n_classes_part_1 = np.unique(labels_true_part_1).shape[0]
    n_clusters_part_1 = np.unique(labels_pred_part_1).shape[0]
    n_classes_part_2 = np.unique(labels_true_part_2).shape[0]
    n_clusters_part_2 = np.unique(labels_pred_part_2).shape[0]

    # Special limit cases: no clustering since the data is not split;
    # or trivial clustering. These are perfect matches hence return 1.0.
    if (
        (
            n_classes_part_1
            == n_clusters_part_1
            == n_classes_part_2
            == n_clusters_part_2
            == 1
        )
        or n_classes_part_1
        == n_clusters_part_1
        == n_classes_part_2
        == n_clusters_part_2
        == 0
        or (
            n_classes_part_1 == n_clusters_part_1 == n_samples_part_1
            and n_classes_part_2 == n_clusters_part_2 == n_samples_part_2
        )
    ):
        return 1.0

    # Compute the ARI using the contingency data
    contingency_part_1 = contingency_matrix(
        labels_true_part_1, labels_pred_part_1, sparse=True
    )
    contingency_part_2 = contingency_matrix(
        labels_true_part_2, labels_pred_part_2, sparse=True
    )

    # Theorem 3.3 of Robert2019 (https://hal.inria.fr/hal-01524832v4) defines
    # the final contingency matrix by the Kronecker product between the two
    # contingency matrices of patition 1 and 2.
    contingency = sp.kron(contingency_part_1, contingency_part_2, format="csr")

    sum_comb_c = sum(
        comb(n_c, 2, exact=1) for n_c in np.ravel(contingency.sum(axis=1))
    )
    sum_comb_k = sum(
        comb(n_k, 2, exact=1) for n_k in np.ravel(contingency.sum(axis=0))
    )
    sum_comb = sum(comb(n_ij, 2, exact=1) for n_ij in contingency.data)

    prod_comb = (sum_comb_c * sum_comb_k) / comb(
        n_samples_part_1 * n_samples_part_2, 2, exact=1
    )
    mean_comb = (sum_comb_k + sum_comb_c) / 2.0
    return (sum_comb - prod_comb) / (mean_comb - prod_comb)
