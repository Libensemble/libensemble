"""Persistent generator providing points using sampling"""

import numpy as np

from libensemble.message_numbers import EVAL_GEN_TAG, FINISHED_PERSISTENT_GEN_TAG, PERSIS_STOP, STOP_TAG
from libensemble.specs import output_data, persistent_input_fields
from libensemble.tools.persistent_support import PersistentSupport

__all__ = [
    "persistent_uniform",
    "persistent_uniform_final_update",
    "persistent_request_shutdown",
    "uniform_nonblocking",
    "batched_history_matching",
    "persistent_uniform_with_cancellations",
]


def _get_user_params(user_specs):
    """Extract user params"""
    b = user_specs["initial_batch_size"]
    ub = user_specs["ub"]
    lb = user_specs["lb"]
    n = len(lb)  # dimension
    assert isinstance(b, int), "Batch size must be an integer"
    assert isinstance(n, int), "Dimension must be an integer"
    assert isinstance(lb, np.ndarray), "lb must be a numpy array"
    assert isinstance(ub, np.ndarray), "ub must be a numpy array"
    return b, n, lb, ub


@persistent_input_fields(["sim_id"])
@output_data([("x", float, (2,))])  # The dimension of 2 is  a default and can be overwritten
def persistent_uniform(_, persis_info, gen_specs, libE_info):
    """
    This generation function always enters into persistent mode and returns
    ``gen_specs["initial_batch_size"]`` uniformly sampled points the first time it
    is called. Afterwards, it returns the number of points given. This can be
    used in either a batch or asynchronous mode by adjusting the allocation
    function.

    .. seealso::
        `test_persistent_uniform_sampling.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/functionality_tests/test_persistent_uniform_sampling.py>`_
        `test_persistent_uniform_sampling_async.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/functionality_tests/test_persistent_uniform_sampling_async.py>`_
    """  # noqa

    b, n, lb, ub = _get_user_params(gen_specs["user"])
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)

    # Send batches until manager sends stop tag
    tag = None
    while tag not in [STOP_TAG, PERSIS_STOP]:
        H_o = np.zeros(b, dtype=gen_specs["out"])
        H_o["x"] = persis_info["rand_stream"].uniform(lb, ub, (b, n))
        if "obj_component" in H_o.dtype.fields:
            H_o["obj_component"] = persis_info["rand_stream"].integers(
                low=0, high=gen_specs["user"]["num_components"], size=b
            )
        tag, Work, calc_in = ps.send_recv(H_o)
        if hasattr(calc_in, "__len__"):
            b = len(calc_in)

    return None, persis_info, FINISHED_PERSISTENT_GEN_TAG


def persistent_uniform_final_update(_, persis_info, gen_specs, libE_info):
    """
    Assuming the value ``"f"`` returned from sim_f is stochastic, this
    generation is updating an estimated mean ``"f_est"`` of the sim_f output at
    each of the corners of the domain.

    .. seealso::
        `test_persistent_uniform_sampling_running_mean.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/functionality_tests/test_persistent_uniform_sampling_running_mean.py>`_
    """  # noqa

    b, n, lb, ub = _get_user_params(gen_specs["user"])
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)

    def generate_corners(x, y):
        n = len(x)
        corner_indices = np.arange(2**n)
        corners = []
        for index in corner_indices:
            corner = [x[i] if index & (1 << i) else y[i] for i in range(n)]
            corners.append(corner)
        return corners

    def sample_corners_with_probability(corners, p, b):
        selected_corners = np.random.choice(len(corners), size=b, p=p)
        sampled_corners = [corners[i] for i in selected_corners]
        return sampled_corners, selected_corners

    corners = generate_corners(lb, ub)

    # Start with equal probabilities
    p = np.ones(2**n) / 2**n

    running_total = np.nan * np.ones(2**n)
    number_of_samples = np.zeros(2**n)
    sent = np.array([], dtype=int)

    # Send batches of `b` points until manager sends stop tag
    tag = None
    next_id = 0
    while tag not in [STOP_TAG, PERSIS_STOP]:
        H_o = np.zeros(b, dtype=gen_specs["out"])
        H_o["sim_id"] = range(next_id, next_id + b)
        next_id += b

        sampled_corners, corner_ids = sample_corners_with_probability(corners, p, b)

        H_o["corner_id"] = corner_ids
        H_o["x"] = sampled_corners
        sent = np.append(sent, corner_ids)

        tag, Work, calc_in = ps.send_recv(H_o)
        if hasattr(calc_in, "__len__"):
            b = len(calc_in)
            for row in calc_in:
                number_of_samples[row["corner_id"]] += 1
                if np.isnan(running_total[row["corner_id"]]):
                    running_total[row["corner_id"]] = row["f"]
                else:
                    running_total[row["corner_id"]] += row["f"]

    # Having received a PERSIS_STOP, update f_est field for all points and return
    # For manager to honor final H_o return, must have set libE_specs["use_persis_return_gen"] = True
    f_est = running_total / number_of_samples
    H_o = np.zeros(len(sent), dtype=[("sim_id", int), ("corner_id", int), ("f_est", float)])
    for count, i in enumerate(sent):
        H_o["sim_id"][count] = count
        H_o["corner_id"][count] = i
        H_o["f_est"][count] = f_est[i]

    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG


def persistent_request_shutdown(_, persis_info, gen_specs, libE_info):
    """
    This generation function is similar in structure to persistent_uniform,
    but uses a count to test exiting on a threshold value. This principle can
    be used with a supporting allocation function (e.g. start_only_persistent)
    to shutdown an ensemble when a condition is met.

    .. seealso::
        `test_persistent_uniform_gen_decides_stop.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/functionality_tests/test_persistent_uniform_gen_decides_stop.py>`_
    """  # noqa
    b, n, lb, ub = _get_user_params(gen_specs["user"])
    shutdown_limit = gen_specs["user"]["shutdown_limit"]
    f_count = 0
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)

    # Send batches until manager sends stop tag
    tag = None
    while tag not in [STOP_TAG, PERSIS_STOP]:
        H_o = np.zeros(b, dtype=gen_specs["out"])
        H_o["x"] = persis_info["rand_stream"].uniform(lb, ub, (b, n))
        tag, Work, calc_in = ps.send_recv(H_o)
        if hasattr(calc_in, "__len__"):
            b = len(calc_in)
        f_count += b
        if f_count >= shutdown_limit:
            print("Reached threshold.", f_count, flush=True)
            break  # End the persistent gen

    return None, persis_info, FINISHED_PERSISTENT_GEN_TAG


def uniform_nonblocking(_, persis_info, gen_specs, libE_info):
    """
    This generation function is designed to test non-blocking receives.

    .. seealso::
        `test_persistent_uniform_sampling.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/functionality_tests/test_persistent_uniform_sampling.py>`_
    """  # noqa
    b, n, lb, ub = _get_user_params(gen_specs["user"])
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)

    # Send batches until manager sends stop tag
    tag = None
    while tag not in [STOP_TAG, PERSIS_STOP]:
        H_o = np.zeros(b, dtype=gen_specs["out"])
        H_o["x"] = persis_info["rand_stream"].uniform(lb, ub, (b, n))
        ps.send(H_o)

        received = False
        spin_count = 0
        while not received:
            tag, Work, calc_in = ps.recv(blocking=False)
            if tag is not None:
                received = True
            else:
                spin_count += 1

        persis_info["spin_count"] = spin_count

        if hasattr(calc_in, "__len__"):
            b = len(calc_in)

    return None, persis_info, FINISHED_PERSISTENT_GEN_TAG


def batched_history_matching(_, persis_info, gen_specs, libE_info):
    """
    Given
    - sim_f with an input of x with len(x)=n
    - b, the batch size of points to generate
    - q<b, the number of best samples to use in the following iteration

    Pseudocode:
    Let (mu, Sigma) denote a mean and covariance matrix initialized to the
    origin and the identity, respectively.

    While true (batch synchronous for now):

        Draw b samples x_1, ... , x_b from MVN( mu, Sigma)
        Evaluate f(x_1), ... , f(x_b) and determine the set of q x_i whose f(x_i) values are smallest (breaking ties lexicographically)
        Update (mu, Sigma) based on the sample mean and sample covariance of these q x values.

    .. seealso::
        `test_persistent_uniform_sampling.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/functionality_tests/test_persistent_uniform_sampling.py>`_
    """  # noqa
    lb = gen_specs["user"]["lb"]

    n = len(lb)
    b = gen_specs["user"]["initial_batch_size"]
    q = gen_specs["user"]["num_best_vals"]
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)

    mu = np.zeros(n)
    Sigma = np.eye(n)
    tag = None

    while tag not in [STOP_TAG, PERSIS_STOP]:
        H_o = np.zeros(b, dtype=gen_specs["out"])
        H_o["x"] = persis_info["rand_stream"].multivariate_normal(mu, Sigma, b)

        # Send data and get next assignment
        tag, Work, calc_in = ps.send_recv(H_o)
        if calc_in is not None:
            all_inds = np.argsort(calc_in["f"])
            best_inds = all_inds[:q]
            mu = np.mean(H_o["x"][best_inds], axis=0)
            Sigma = np.cov(H_o["x"][best_inds].T)

    return None, persis_info, FINISHED_PERSISTENT_GEN_TAG


def persistent_uniform_with_cancellations(_, persis_info, gen_specs, libE_info):
    ub = gen_specs["user"]["ub"]
    lb = gen_specs["user"]["lb"]
    n = len(lb)
    b = gen_specs["user"]["initial_batch_size"]

    # Start cancelling points from half initial batch onward
    cancel_from = b // 2  # Should get at least this many points back

    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)

    # Send batches until manager sends stop tag
    tag = None
    while tag not in [STOP_TAG, PERSIS_STOP]:
        H_o = np.zeros(b, dtype=gen_specs["out"])
        H_o["x"] = persis_info["rand_stream"].uniform(lb, ub, (b, n))
        tag, Work, calc_in = ps.send_recv(H_o)

        if hasattr(calc_in, "__len__"):
            b = len(calc_in)

            # Cancel as many points as got back
            cancel_ids = list(range(cancel_from, cancel_from + b))
            cancel_from += b
            ps.request_cancel_sim_ids(cancel_ids)

    return None, persis_info, FINISHED_PERSISTENT_GEN_TAG
