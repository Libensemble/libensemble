import numpy as np

from libensemble.tools.tools import add_unique_random_streams
from libensemble.utils.misc import list_dicts_to_np


def _check_conversion(H, npp):

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


def test_asktell_sampling_and_utils():
    from libensemble.gen_classes.sampling import UniformSample

    persis_info = add_unique_random_streams({}, 5, seed=1234)
    gen_specs = {
        "out": [("x", float, (2,))],
        "user": {
            "lb": np.array([-3, -2]),
            "ub": np.array([3, 2]),
        },
    }

    # Test initialization with libensembley parameters
    gen = UniformSample(None, persis_info[1], gen_specs, None)
    assert len(gen.ask(10)) == 10

    # Test initialization gen-specific keyword args
    gen = UniformSample(gen_specs=gen_specs, lb=np.array([-3, -2]), ub=np.array([3, 2]))
    assert len(gen.ask(10)) == 10

    out_np = gen.ask_numpy(3)  # should get numpy arrays, non-flattened
    out = gen.ask(3)  # needs to get dicts, 2d+ arrays need to be flattened
    assert all([len(x) == 2 for x in out])  # np_to_list_dicts is now tested

    # now we test list_dicts_to_np directly
    out_np = list_dicts_to_np(out)

    # check combined values resemble flattened list-of-dicts values
    assert out_np.dtype.names == ("x",)
    for i, entry in enumerate(out):
        for j, value in enumerate(entry.values()):
            assert value == out_np["x"][i][j]


def test_awkward_list_dict():
    from libensemble.utils.misc import list_dicts_to_np

    # test list_dicts_to_np on a weirdly formatted dictionary
    # Unfortunately, we're not really checking against some original
    #  libE-styled source of truth, like H.

    weird_list_dict = [
        {
            "x0": "abcd",
            "x1": "efgh",
            "y": 56,
            "z0": 1,
            "z1": 2,
            "z2": 3,
            "z3": 4,
            "z4": 5,
            "z5": 6,
            "z6": 7,
            "z7": 8,
            "z8": 9,
            "z9": 10,
            "z10": 11,
            "a0": "B",
        }
    ]

    out_np = list_dicts_to_np(weird_list_dict)

    assert all([i in ("x", "y", "z", "a0") for i in out_np.dtype.names])


def test_awkward_H():
    from libensemble.utils.misc import list_dicts_to_np, np_to_list_dicts

    dtype = [("a", "i4"), ("x", "f4", (3,)), ("y", "f4", (1,)), ("z", "f4", (12,)), ("greeting", "U10"), ("co2", "f8")]
    H = np.zeros(2, dtype=dtype)
    H[0] = (1, [1.1, 2.2, 3.3], [10.1], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "hello", "1.23")
    H[1] = (2, [4.4, 5.5, 6.6], [11.1], [51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62], "goodbye", "2.23")

    list_dicts = np_to_list_dicts(H)
    npp = list_dicts_to_np(list_dicts, dtype=dtype)
    _check_conversion(H, npp)


if __name__ == "__main__":
    test_asktell_sampling_and_utils()
    test_awkward_list_dict()
    test_awkward_H()
