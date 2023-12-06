"""Persistent generator providing points using sampling"""

import numpy as np
from gpcam import GPOptimizer as GP

from libensemble.message_numbers import EVAL_GEN_TAG, FINISHED_PERSISTENT_GEN_TAG, PERSIS_STOP, STOP_TAG
from libensemble.tools.persistent_support import PersistentSupport

__all__ = [
    "persistent_gpCAM",
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


def persistent_gpCAM(H_in, persis_info, gen_specs, libE_info):
    """
    This generation function always enters into persistent mode and returns
    ``gen_specs["initial_batch_size"]`` uniformly sampled points the first time it
    is called. Afterwards, it returns the number of points given. This can be
    used in either a batch or asynchronous mode by adjusting the allocation
    function.

    .. seealso::
        `test_gpCAM.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_gpCAM.py>`_
    """  # noqa

    batch_size, n, lb, ub = _get_user_params(gen_specs["user"])
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)

    all_x = np.empty((0, n))
    all_y = np.empty((0, 1))

    hps_bounds = np.vstack(([0.001, 1e9], np.tile([1e-3, 1e2], (n, 1))))

    # Send batches until manager sends stop tag
    tag = None
    while tag not in [STOP_TAG, PERSIS_STOP]:

        if all_x.shape[0] == 0:
            x_new = persis_info["rand_stream"].uniform(lb, ub, (batch_size, n))
        else:
            init_hps = np.random.uniform(size=len(hps_bounds), low=hps_bounds[:, 0], high=hps_bounds[:, 1])
            # We are assuming deterministic y, so we set the noise to be tiny
            my_gp2S = GP(all_x, all_y, init_hps, noise_variances=1e-8 * np.ones(len(all_y)))

            my_gp2S.train(hps_bounds, max_iter=2)

            x_for_var = persis_info["rand_stream"].uniform(lb, ub, (10 * batch_size, n))
            var_rand = my_gp2S.posterior_covariance(x_for_var, variance_only=True)["v(x)"]
            inds_for_largest_var = np.argsort(var_rand)[-batch_size:]
            x_new = x_for_var[inds_for_largest_var]

        H_o = np.zeros(batch_size, dtype=gen_specs["out"])
        H_o["x"] = x_new
        tag, Work, calc_in = ps.send_recv(H_o)
        if not hasattr(calc_in, "__len__"):
            break

        all_x = np.vstack((all_x, x_new))
        all_y = np.vstack((all_y, np.atleast_2d(calc_in["f"]).T))

    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG
