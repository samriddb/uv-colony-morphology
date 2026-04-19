import numpy as np
import pandas as pd
from scipy.spatial import KDTree


def knn(df, k=5):
    """kdtree over colony centroids - returns distances, indices, and summary df"""

    if len(df) < 2:
        return None, None, None

    k = min(k, len(df) - 1)
    coords = df[["cy", "cx"]].values

    tree = KDTree(coords)
    dists, idxs = tree.query(coords, k=k + 1)  # +1 because self is included

    # drop self (col 0)
    dists = dists[:, 1:]
    idxs  = idxs[:, 1:]

    summary = pd.DataFrame({
        "label":        df["label"].values,
        "nn1_dist":     dists[:, 0],
        "mean_k_dist":  dists.mean(axis=1),
        "std_k_dist":   dists.std(axis=1),
    })

    return dists, idxs, summary
