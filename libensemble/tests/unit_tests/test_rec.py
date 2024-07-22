import numpy as np
import scipy
from scipy.spatial.distance import cdist

def test_cdist():

    print(f"NumPy version: {np.__version__}")
    print(f"SciPy version: {scipy.__version__}")

    a = [[0.0, 0.0]]
    b = [[0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0]]

    distances = cdist(a, b, "euclidean")
    print(distances)

test_cdist()
