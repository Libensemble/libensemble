import numpy as np
from scipy.spatial.distance import cdist


def test_cdist_issue():
    """There is an issue (at least in scipy 1.1.0) with cdist segfaulting."""

    H = np.zeros(20, dtype=[('x', '<f8', (2,)), ('m', '<i8'), ('a', '<f8'), ('b', '?'), ('c', '?'), ('d', '<f8'), ('e', '<f8'), ('fa', '<f8'), ('g', '<i8'), ('h', '<i8'), ('i', '?'), ('j', '<i8'), ('k', '?'), ('f', '<f8'), ('l', '?')])
    np.random.seed(1)
    H['x'] = np.random.uniform(0, 1, (20, 2))
    dist_1 = cdist(np.atleast_2d(H['x'][3]), H['x'], 'euclidean')
    assert len(dist_1), "We didn't segfault"
