import numpy as np

from libensemble.tools.tools import add_unique_random_streams


def test_asktell_sampling():
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

    import ipdb

    ipdb.set_trace()

    out = gen.ask_numpy(3)  # should get numpy arrays, non-flattened
    out = gen.ask(3)  # needs to get dicts, 2d+ arrays need to be flattened
    assert all([len(x) == 2 for x in out])  # np_to_list_dicts is now tested


if __name__ == "__main__":
    test_asktell_sampling()
