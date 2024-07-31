import numpy as np
import pytest

from libensemble.gen_funcs.persistent_aposmm import cdist

# @pytest.mark.extra
# def test_cdist_issue():
#     try:
#         from scipy.spatial.distance import cdist
#     except ModuleNotFoundError:
#         pytest.skip("scipy or its dependencies not importable. Skipping.")
#
#     """There is an issue (at least in scipy 1.1.0) with cdist segfaulting."""
#     H = np.zeros(
#         20,
#         dtype=[
#             ("x", "<f8", (2,)),
#             ("m", "<i8"),
#             ("a", "<f8"),
#             ("b", "?"),
#             ("c", "?"),
#             ("d", "<f8"),
#             ("e", "<f8"),
#             ("fa", "<f8"),
#             ("g", "<i8"),
#             ("h", "<i8"),
#             ("i", "?"),
#             ("j", "<i8"),
#             ("k", "?"),
#             ("f", "<f8"),
#             ("l", "?"),
#         ],
#     )
#     np.random.seed(1)
#     H["x"] = np.random.uniform(0, 1, (20, 2))
#     dist_1 = cdist(np.atleast_2d(H["x"][3]), H["x"], "euclidean")
#     assert len(dist_1), "We didn't segfault"


def test_cdist_same_size():
    XA = np.array([[1, 2], [3, 4], [5, 6]])
    XB = np.array([[7, 8], [9, 10], [11, 12]])
    exp = np.array(
        [
            [8.48528137, 11.3137085, 14.1421356],
            [5.65685425, 8.48528137, 11.3137085],
            [2.82842712, 5.65685425, 8.48528137],
        ]
    )

    result = cdist(XA, XB)
    assert np.allclose(result, exp), f"Result: {result}, Expected: {exp}"


def test_cdist_different_size():
    XA = np.array([[1, 2], [3, 4]])
    XB = np.array([[5, 6], [7, 8], [9, 10]])
    exp = np.array(
        [
            [5.65685425, 8.48528137, 11.3137085],
            [2.82842712, 5.65685425, 8.48528137],
        ]
    )

    result = cdist(XA, XB)
    assert np.allclose(result, exp), f"Result: {result}, Expected: {exp}"


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
    # test_cdist_issue()
    test_cdist_same_size()
    test_cdist_different_size()
    test_save()
