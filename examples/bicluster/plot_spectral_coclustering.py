"""
==============================================
A demo of the Spectral Co-Clustering algorithm
==============================================

This example demonstrates how to generate a dataset and bicluster it
using the Spectral Co-Clustering algorithm.

The dataset is generated using the ``make_biclusters`` function, which
creates a matrix of small values and implants bicluster with large
values. The rows and columns are then shuffled and passed to the
Spectral Co-Clustering algorithm. Rearranging the shuffled matrix to
make biclusters contiguous shows how accurately the algorithm found
the biclusters.

"""
print(__doc__)

# Author: Kemal Eren <kemal@kemaleren.com>
# License: BSD 3 clause

import numpy as np  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

from sklearn.datasets import make_biclusters  # noqa: E402
from sklearn.cluster import SpectralCoclustering  # noqa: E402
from sklearn.metrics import (
    consensus_score,
    coclustering_adjusted_rand_score,
)  # noqa: E402

data, rows, columns = make_biclusters(
    shape=(300, 300), n_clusters=5, noise=5, shuffle=False, random_state=0
)

plt.matshow(data, cmap=plt.cm.Blues)
plt.title("Original dataset")

# shuffle clusters
rng = np.random.RandomState(0)
row_idx = rng.permutation(data.shape[0])
col_idx = rng.permutation(data.shape[1])
data = data[row_idx][:, col_idx]

plt.matshow(data, cmap=plt.cm.Blues)
plt.title("Shuffled dataset")

model = SpectralCoclustering(n_clusters=5, random_state=0)
model.fit(data)
consensus_score = consensus_score(
    model.biclusters_, (rows[:, row_idx], columns[:, col_idx])
)
cari_score = coclustering_adjusted_rand_score(
    model.biclusters_[0].argmax(axis=0),
    model.biclusters_[1].argmax(axis=0),
    rows[:, row_idx].argmax(axis=0),
    columns[:, col_idx].argmax(axis=0),
)
print("consensus score: {:.3f}".format(consensus_score))
print("coclassification adjusted rand score: {:.3f}".format(cari_score))
fit_data = data[np.argsort(model.row_labels_)]
fit_data = fit_data[:, np.argsort(model.column_labels_)]

plt.matshow(fit_data, cmap=plt.cm.Blues)
plt.title("After biclustering; rearranged to show biclusters")

plt.show()
