import numpy as np
from libensemble.utils.misc import unmap_numpy_array


def _check_conversion(H, npp, mapping={}):

    for field in H.dtype.names:
        print(f"Comparing {field}: {H[field]} {npp[field]}")

        if isinstance(H[field], np.ndarray):
            assert np.array_equal(H[field], npp[field]), f"Mismatch found in field {field}"

        elif isinstance(H[field], str) and isinstance(npp[field], str):
            assert H[field] == npp[field], f"Mismatch found in field {field}"

        elif np.isscalar(H[field]) and np.isscalar(npp[field]):
            assert np.isclose(H[field], npp[field]), f"Mismatch found in field {field}"

        else:
            raise TypeError(f"Unhandled or mismatched types in field {field}: {type(H[field])} vs {type(npp[field])}")


# def test_awkward_list_dict():
#     from libensemble.utils.misc import list_dicts_to_np

#     # test list_dicts_to_np on a weirdly formatted dictionary
#     # Unfortunately, we're not really checking against some original
#     #  libE-styled source of truth, like H.

#     weird_list_dict = [
#         {
#             "x0": "abcd",
#             "x1": "efgh",
#             "y": 56,
#             "z0": 1,
#             "z1": 2,
#             "z2": 3,
#             "z3": 4,
#             "z4": 5,
#             "z5": 6,
#             "z6": 7,
#             "z7": 8,
#             "z8": 9,
#             "z9": 10,
#             "z10": 11,
#             "a0": "B",
#         }
#     ]

#     out_np = list_dicts_to_np(weird_list_dict)

#     assert all([i in ("x", "y", "z", "a0") for i in out_np.dtype.names])

#     weird_list_dict = [
#         {
#             "sim_id": 77,
#             "core": 89,
#             "edge": 10.1,
#             "beam": 76.5,
#             "energy": 12.34,
#             "local_pt": True,
#             "local_min": False,
#         },
#         {
#             "sim_id": 10,
#             "core": 32.8,
#             "edge": 16.2,
#             "beam": 33.5,
#             "energy": 99.34,
#             "local_pt": False,
#             "local_min": False,
#         },
#     ]

#     # target dtype: [("sim_id", int), ("x, float, (3,)), ("f", float), ("local_pt", bool), ("local_min", bool)]

#     mapping = {"x": ["core", "edge", "beam"], "f": ["energy"]}
#     out_np = list_dicts_to_np(weird_list_dict, mapping=mapping)

#     assert all([i in ("sim_id", "x", "f", "local_pt", "local_min") for i in out_np.dtype.names])


def test_awkward_H():
    from libensemble.utils.misc import list_dicts_to_np, np_to_list_dicts

    dtype = [("a", "i4"), ("x", "f4", (3,)), ("y", "f4", (1,)), ("z", "f4", (12,)), ("greeting", "U10"), ("co2", "f8")]
    H = np.zeros(2, dtype=dtype)
    H[0] = (1, [1.1, 2.2, 3.3], [10.1], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "hello", "1.23")
    H[1] = (2, [4.4, 5.5, 6.6], [11.1], [51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62], "goodbye", "2.23")

    list_dicts = np_to_list_dicts(H)
    npp = list_dicts_to_np(list_dicts, dtype=dtype)
    _check_conversion(H, npp)


def test_unmap_numpy_array_basic():
    """Test basic unmapping of x and x_on_cube arrays"""

    dtype = [("sim_id", int), ("x", float, (3,)), ("x_on_cube", float, (3,)), ("f", float), ("grad", float, (3,))]
    H = np.zeros(2, dtype=dtype)
    H[0] = (0, [1.1, 2.2, 3.3], [0.1, 0.2, 0.3], 10.5, [0.1, 0.2, 0.3])
    H[1] = (1, [4.4, 5.5, 6.6], [0.4, 0.5, 0.6], 20.7, [0.4, 0.5, 0.6])

    mapping = {"x": ["x0", "x1", "x2"], "x_on_cube": ["x0_cube", "x1_cube", "x2_cube"]}
    H_unmapped = unmap_numpy_array(H, mapping)

    expected_fields = ["sim_id", "x0", "x1", "x2", "x0_cube", "x1_cube", "x2_cube", "f"]
    assert all(field in H_unmapped.dtype.names for field in expected_fields)

    assert H_unmapped["x0"][0] == 1.1
    assert H_unmapped["x1"][0] == 2.2
    assert H_unmapped["x2"][0] == 3.3
    assert H_unmapped["x0_cube"][0] == 0.1
    assert H_unmapped["x1_cube"][0] == 0.2
    assert H_unmapped["x2_cube"][0] == 0.3
    # Test that non-mapped array fields are passed through unchanged
    assert "grad" in H_unmapped.dtype.names
    assert np.array_equal(H_unmapped["grad"], H["grad"])


def test_unmap_numpy_array_single_dimension():
    """Test unmapping with single dimension"""

    dtype = [("sim_id", int), ("x", float, (1,)), ("f", float)]
    H = np.zeros(1, dtype=dtype)
    H[0] = (0, [5.5], 15.0)

    mapping = {"x": ["x0"]}
    H_unmapped = unmap_numpy_array(H, mapping)

    assert "x0" in H_unmapped.dtype.names
    assert H_unmapped["x0"][0] == 5.5


def test_unmap_numpy_array_edge_cases():
    """Test edge cases for unmap_numpy_array"""

    dtype = [("sim_id", int), ("x", float, (2,)), ("f", float)]
    H = np.zeros(1, dtype=dtype)
    H[0] = (0, [1.0, 2.0], 10.0)

    # No mapping
    H_no_mapping = unmap_numpy_array(H, {})
    assert H_no_mapping is H

    # None array
    H_none = unmap_numpy_array(None, {"x": ["x0", "x1"]})
    assert H_none is None


if __name__ == "__main__":
    # test_awkward_list_dict()
    test_awkward_H()
    test_unmap_numpy_array_basic()
    test_unmap_numpy_array_single_dimension()
    test_unmap_numpy_array_edge_cases()
