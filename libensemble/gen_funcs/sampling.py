"""
This module contains multiple generator functions for sampling a domain.
"""

import numpy as np

from libensemble.specs import output_data

__all__ = [
    "uniform_random_sample",
    "uniform_random_sample_with_variable_resources",
    "uniform_random_sample_with_var_priorities_and_resources",
    "uniform_random_sample_obj_components",
    "latin_hypercube_sample",
    "uniform_random_sample_cancel",
]


@output_data([("x", float, 2)])  # default: can be overwritten in gen_specs
def uniform_random_sample(_, persis_info, gen_specs):
    """
    Generates ``gen_specs["user"]["gen_batch_size"]`` points uniformly over the domain
    defined by ``gen_specs["user"]["ub"]`` and ``gen_specs["user"]["lb"]``.

    .. seealso::
        `test_uniform_sampling.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/functionality_tests/test_uniform_sampling.py>`_ # noqa
    """
    ub = gen_specs["user"]["ub"]
    lb = gen_specs["user"]["lb"]

    n = len(lb)
    b = gen_specs["user"]["gen_batch_size"]

    H_o = np.zeros(b, dtype=gen_specs["out"])

    H_o["x"] = persis_info["rand_stream"].uniform(lb, ub, (b, n))

    return H_o, persis_info


def uniform_random_sample_with_variable_resources(_, persis_info, gen_specs):
    """
    Generates ``gen_specs["user"]["gen_batch_size"]`` points uniformly over the domain
    defined by ``gen_specs["user"]["ub"]`` and ``gen_specs["user"]["lb"]``.

    Also randomly requests a different number of resource sets to be used in each evaluation.

    This generator is used to test/demonstrate setting of resource sets.

    .. seealso::
        `test_uniform_sampling_with_variable_resources.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/functionality_tests/test_uniform_sampling_with_variable_resources.py>`_ # noqa
    """

    ub = gen_specs["user"]["ub"]
    lb = gen_specs["user"]["lb"]
    max_rsets = gen_specs["user"]["max_resource_sets"]

    n = len(lb)
    b = gen_specs["user"]["gen_batch_size"]

    H_o = np.zeros(b, dtype=gen_specs["out"])

    H_o["x"] = persis_info["rand_stream"].uniform(lb, ub, (b, n))
    H_o["resource_sets"] = persis_info["rand_stream"].integers(1, max_rsets + 1, b)

    print(f'GEN: H rsets requested: {H_o["resource_sets"]}')

    return H_o, persis_info


def uniform_random_sample_with_var_priorities_and_resources(H, persis_info, gen_specs):
    """
    Generates points uniformly over the domain defined by ``gen_specs["user"]["ub"]`` and
    ``gen_specs["user"]["lb"]``. Also, randomly requests a different priority and number of
    resource sets to be used in the evaluation of the generated points, after the initial batch.

    This generator is used to test/demonstrate setting of priorities and resource sets.

    """
    ub = gen_specs["user"]["ub"]
    lb = gen_specs["user"]["lb"]
    max_rsets = gen_specs["user"]["max_resource_sets"]

    n = len(lb)

    if len(H) == 0:
        b = gen_specs["user"]["initial_batch_size"]

        H_o = np.zeros(b, dtype=gen_specs["out"])
        for i in range(0, b):
            # x= i*np.ones(n)
            x = persis_info["rand_stream"].uniform(lb, ub, (1, n))
            H_o["x"][i] = x
            H_o["resource_sets"][i] = 1
            H_o["priority"] = 1

    else:
        H_o = np.zeros(1, dtype=gen_specs["out"])
        # H_o["x"] = len(H)*np.ones(n)  # Can use a simple count for testing.
        H_o["x"] = persis_info["rand_stream"].uniform(lb, ub)
        H_o["resource_sets"] = persis_info["rand_stream"].integers(1, max_rsets + 1)
        H_o["priority"] = 10 * H_o["resource_sets"]
        # print("Created sim for {} resource sets".format(H_o["resource_sets"]), flush=True)

    return H_o, persis_info


def uniform_random_sample_obj_components(H, persis_info, gen_specs):
    """
    Generates points uniformly over the domain defined by ``gen_specs["user"]["ub"]``
    and ``gen_specs["user"]["lb"]`` but requests each ``obj_component`` be evaluated
    separately.

    .. seealso::
        `test_uniform_sampling_one_residual_at_a_time.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/functionality_tests/test_uniform_sampling_one_residual_at_a_time.py>`_ # noqa
    """
    ub = gen_specs["user"]["ub"]
    lb = gen_specs["user"]["lb"]

    n = len(lb)
    m = gen_specs["user"]["components"]
    b = gen_specs["user"]["gen_batch_size"]

    H_o = np.zeros(b * m, dtype=gen_specs["out"])
    for i in range(0, b):
        x = persis_info["rand_stream"].uniform(lb, ub, (1, n))
        H_o["x"][i * m : (i + 1) * m, :] = np.tile(x, (m, 1))
        H_o["priority"][i * m : (i + 1) * m] = persis_info["rand_stream"].uniform(0, 1, m)
        H_o["obj_component"][i * m : (i + 1) * m] = np.arange(0, m)

        H_o["pt_id"][i * m : (i + 1) * m] = len(H) // m + i

    return H_o, persis_info


def uniform_random_sample_cancel(_, persis_info, gen_specs):
    """
    Similar to uniform_random_sample but with immediate cancellation of
    selected points for testing.

    """
    ub = gen_specs["user"]["ub"]
    lb = gen_specs["user"]["lb"]

    n = len(lb)
    b = gen_specs["user"]["gen_batch_size"]

    H_o = np.zeros(b, dtype=gen_specs["out"])
    for i in range(b):
        if i % 10 == 0:
            H_o[i]["cancel_requested"] = True

    H_o["x"] = persis_info["rand_stream"].uniform(lb, ub, (b, n))

    return H_o, persis_info


@output_data([("x", float, (1,))])
def latin_hypercube_sample(_, persis_info, gen_specs):
    """
    Generates ``gen_specs["user"]["gen_batch_size"]`` points in a Latin
    hypercube sample over the domain defined by ``gen_specs["user"]["ub"]`` and
    ``gen_specs["user"]["lb"]``.

    .. seealso::
        `test_1d_sampling.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_1d_sampling.py>`_ # noqa
    """

    ub = gen_specs["user"]["ub"]
    lb = gen_specs["user"]["lb"]

    n = len(lb)
    b = gen_specs["user"]["gen_batch_size"]

    H_o = np.zeros(b, dtype=gen_specs["out"])

    A = lhs_sample(n, b, persis_info["rand_stream"])

    H_o["x"] = A * (ub - lb) + lb

    return H_o, persis_info


def lhs_sample(n, k, stream):
    # Generate the intervals and random values
    intervals = np.linspace(0, 1, k + 1)
    rand_source = stream.uniform(0, 1, (k, n))
    rand_pts = np.zeros((k, n))
    sample = np.zeros((k, n))

    # Add a point uniformly in each interval
    a = intervals[:k]
    b = intervals[1:]
    for j in range(n):
        rand_pts[:, j] = rand_source[:, j] * (b - a) + a

    # Randomly perturb
    for j in range(n):
        sample[:, j] = rand_pts[stream.permutation(k), j]

    return sample
