import numpy as np
from scipy.spatial.distance import cdist

def test_cdist():
    a = [[0.0, 0.0]]
    b = [[0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0]]

    distances = cdist(a, b, "euclidean")
    print(distances)

test_cdist()
