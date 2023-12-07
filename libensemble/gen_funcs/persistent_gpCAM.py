"""Persistent generator exposing gpCAM functionality"""

import time

import numpy as np
from gpcam import GPOptimizer as GP

from libensemble.message_numbers import EVAL_GEN_TAG, FINISHED_PERSISTENT_GEN_TAG, PERSIS_STOP, STOP_TAG
from libensemble.tools.persistent_support import PersistentSupport

__all__ = [
    "persistent_gpCAM_simple",
    "persistent_gpCAM_ask_tell",
]


def _get_user_params(user_specs):
    """Extract user params"""
    b = user_specs["batch_size"]
    ub = user_specs["ub"]
    lb = user_specs["lb"]
    n = len(lb)  # dimension
    assert isinstance(b, int), "Batch size must be an integer"
    assert isinstance(n, int), "Dimension must be an integer"
    assert isinstance(lb, np.ndarray), "lb must be a numpy array"
    assert isinstance(ub, np.ndarray), "ub must be a numpy array"
    return b, n, lb, ub


def persistent_gpCAM_simple(H_in, persis_info, gen_specs, libE_info):
    """
    This generation function constructs a global surrogate of `f` values.
    It is a batched method that produces a first batch uniformly random from
    (lb, ub) and on following iterations samples the GP posterior covariance
    function to find sample points.

    .. seealso::
        `test_gpCAM.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_gpCAM.py>`_
    """  # noqa

    batch_size, n, lb, ub = _get_user_params(gen_specs["user"])
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)

    all_x = np.empty((0, n))
    all_y = np.empty((0, 1))

    # Send batches until manager sends stop tag
    tag = None
    while tag not in [STOP_TAG, PERSIS_STOP]:
        if all_x.shape[0] == 0:
            x_new = persis_info["rand_stream"].uniform(lb, ub, (batch_size, n))
        else:
            # We are assuming deterministic y, so we set the noise to be tiny
            my_gp2S = GP(all_x, all_y, noise_variances=1e-8 * np.ones(len(all_y)))

            my_gp2S.train(max_iter=2)

            x_for_var = persis_info["rand_stream"].uniform(lb, ub, (10 * batch_size, n))
            var_rand = my_gp2S.posterior_covariance(x_for_var, variance_only=True)["v(x)"]
            x_new = x_for_var[np.argsort(var_rand)[-batch_size:]]

        H_o = np.zeros(batch_size, dtype=gen_specs["out"])
        H_o["x"] = x_new
        tag, Work, calc_in = ps.send_recv(H_o)

    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG


def persistent_gpCAM_ask_tell(H_in, persis_info, gen_specs, libE_info):
    """
    Like persistent_gpCAM_simple, this generation function constructs a global
    surrogate of `f` values. It also aa batched method that produces a first batch
    uniformly random from (lb, ub). On subequent iterations, it calls an
    optimization method to produce the next batch of points. This optimization
    might be too slow (relative to the simulation evaluation time) for some use cases.

    .. seealso::
        `test_gpCAM.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_gpCAM.py>`_
    """  # noqa

    batch_size, n, lb, ub = _get_user_params(gen_specs["user"])
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)

    H_o = np.zeros(batch_size, dtype=gen_specs["out"])
    x_new = persis_info["rand_stream"].uniform(lb, ub, (batch_size, n))
    H_o["x"] = x_new

    all_x = np.empty((0, n))
    all_y = np.empty((0, 1))

    tag, Work, calc_in = ps.send_recv(H_o)

    first_call = True
    while tag not in [STOP_TAG, PERSIS_STOP]:
        all_x = np.vstack((all_x, x_new))
        all_y = np.vstack((all_y, np.atleast_2d(calc_in["f"]).T))

        if first_call:
            # Initialize GP
            my_gp2S = GP(all_x, all_y, noise_variances=1e-8 * np.ones(len(all_y)))
        else:
            my_gp2S.tell(all_x, all_y, variances=1e-8)

        my_gp2S.train()

        start = time.time()
        x_new = my_gp2S.ask(
            bounds=np.column_stack((lb, ub)),
            n=batch_size,
            pop_size=batch_size,
            max_iter=1,
        )["x"]
        print(f"Ask time:{time.time() - start}")
        H_o = np.zeros(batch_size, dtype=gen_specs["out"])
        H_o["x"] = x_new

        tag, Work, calc_in = ps.send_recv(H_o)

    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG
