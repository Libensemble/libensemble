import numpy as np
import sys
import scipy
from scipy.spatial.distance import cdist

def test_cdist():

    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"SciPy version: {scipy.__version__}")

    a = [[0.0, 0.0]]
    b = [[0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0]]

    print(f"a: {a}")
    print(f"b: {b}")

    distances = cdist(a, b, "euclidean")
    print(distances)

test_cdist()
