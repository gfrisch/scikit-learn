"""
=============================================
A demo of the Spectral Biclustering algorithm
=============================================

This example demonstrates how to generate a checkerboard dataset and
bicluster it using the Spectral Biclustering algorithm.

The data is generated with the ``make_checkerboard`` function, then
shuffled and passed to the Spectral Biclustering algorithm. The rows
and columns of the shuffled matrix are rearranged to show the
biclusters found by the algorithm.

The outer product of the row and column label vectors shows a
representation of the checkerboard structure.

"""
print(__doc__)

# Author: Kemal Eren <kemal@kemaleren.com>
# License: BSD 3 clause

import numpy as np  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from sklearn.datasets import make_checkerboard  # noqa: E402
from sklearn.cluster import SpectralBiclustering  # noqa: E402
from sklearn.metrics import consensus_score  # noqa: E402
from sklearn.metrics import coclustering_adjusted_rand_score  # noqa: E402


n_clusters = (4, 3)
data, rows, columns = make_checkerboard(
    shape=(300, 300),
    n_clusters=n_clusters,
    noise=10,
    shuffle=False,
    random_state=0,
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

model = SpectralBiclustering(
    n_clusters=n_clusters, method="log", random_state=0
)
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

print("consensus score: {:.1f}".format(consensus_score))
print("coclassification adjusted rand score: {:.3f}".format(cari_score))

fit_data = data[np.argsort(model.row_labels_)]
fit_data = fit_data[:, np.argsort(model.column_labels_)]

plt.matshow(fit_data, cmap=plt.cm.Blues)
plt.title("After biclustering; rearranged to show biclusters")

plt.matshow(
    np.outer(
        np.sort(model.row_labels_) + 1, np.sort(model.column_labels_) + 1
    ),
    cmap=plt.cm.Blues,
)
plt.title("Checkerboard structure of rearranged data")

plt.show()
