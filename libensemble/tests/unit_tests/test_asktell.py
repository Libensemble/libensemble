import numpy as np

from libensemble.tools.tools import add_unique_random_streams
from libensemble.utils.misc import list_dicts_to_np


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
    gen = UniformSample(lb=np.array([-3, -2]), ub=np.array([3, 2]))
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


def test_additional_converts():
    from libensemble.utils.misc import list_dicts_to_np

    # test list_dicts_to_np on a weirdly formatted dictionary
    out_np = list_dicts_to_np(
        [
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
    )

    assert all([i in ("x", "y", "z", "a0") for i in out_np.dtype.names])


if __name__ == "__main__":
    test_asktell_sampling_and_utils()
    test_additional_converts()
