"""Persistent generator providing points using sampling"""

import numpy as np

from libensemble.message_numbers import EVAL_GEN_TAG, FINISHED_PERSISTENT_GEN_TAG, PERSIS_STOP, STOP_TAG
from libensemble.tools.persistent_support import PersistentSupport

__all__ = [
    "persistent_uniform",
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
    return b, n, lb, ub


def persistent_uniform(_, persis_info, gen_specs, libE_info):
    """
    This generation function always enters into persistent mode and returns
    ``gen_specs["initial_batch_size"]`` uniformly sampled points the first time it
    is called. Afterwards, it returns the number of points given. This can be
    used in either a batch or asynchronous mode by adjusting the allocation
    function.

    .. seealso::
        `test_persistent_uniform_sampling.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/functionality_tests/test_persistent_uniform_sampling.py>`_
        `test_persistent_sampling_async.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/functionality_tests/test_persistent_sampling_async.py>`_
    """  # noqa

    b, n, lb, ub = _get_user_params(gen_specs["user"])
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)

    # Send batches until manager sends stop tag
    tag = None
    while tag not in [STOP_TAG, PERSIS_STOP]:
        H_o = np.zeros(b, dtype=gen_specs["out"])
        H_o["x"] = persis_info["rand_stream"].uniform(lb, ub, (b, n))
        tag, Work, calc_in = ps.send_recv(H_o)
        if hasattr(calc_in, "__len__"):
            b = len(calc_in)

    H_o = None
    if gen_specs["user"].get("replace_final_fields", 0):
        # This is only to test libE ability to accept History after a
        # PERSIS_STOP. This history is returned in Work.
        H_o = Work
        H_o["x"] = -1.23

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

    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG


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

    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG


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

    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG


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

    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG
