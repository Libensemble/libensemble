import pytest
import numpy as np


@pytest.mark.extra
def test_cdist_issue():
    try:
        from scipy.spatial.distance import cdist
    except ModuleNotFoundError:
        pytest.skip("scipy or its dependencies not importable. Skipping.")

    """There is an issue (at least in scipy 1.1.0) with cdist segfaulting."""
    H = np.zeros(
        20,
        dtype=[
            ("x", "<f8", (2,)),
            ("m", "<i8"),
            ("a", "<f8"),
            ("b", "?"),
            ("c", "?"),
            ("d", "<f8"),
            ("e", "<f8"),
            ("fa", "<f8"),
            ("g", "<i8"),
            ("h", "<i8"),
            ("i", "?"),
            ("j", "<i8"),
            ("k", "?"),
            ("f", "<f8"),
            ("l", "?"),
        ],
    )
    np.random.seed(1)
    H["x"] = np.random.uniform(0, 1, (20, 2))
    dist_1 = cdist(np.atleast_2d(H["x"][3]), H["x"], "euclidean")
    assert len(dist_1), "We didn't segfault"


@pytest.mark.extra
def test_save():
    """Seeing if I can save parts of the H array."""
    from libensemble.tests.regression_tests.support import uniform_or_localopt_gen_out as gen_out

    n = 2
    gen_out += [("x", float, n), ("x_on_cube", float, n)]
    H = np.zeros(20, dtype=gen_out + [("f", float), ("grad", float, n)])
    np.random.seed(1)
    H["x"] = np.random.uniform(0, 1, (20, 2))
    np.save("H_test", H[["x", "f", "grad"]])

    assert 1, "We saved correctly"


if __name__ == "__main__":
    test_cdist_issue()
    test_save()
