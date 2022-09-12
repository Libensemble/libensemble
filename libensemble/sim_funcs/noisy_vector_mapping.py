"""
This module contains a test noisy function
"""

import numpy as np
from numpy.linalg import norm
from numpy import cos, sin


def func_wrapper(H, persis_info, sim_specs, libE_info):
    """
    Wraps an objective function

    .. seealso::
        `test_persistent_fd_param_finder.py` <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_fd_param_finder.py>`_ # noqa
    """

    batch = len(H["x"])
    H0 = np.zeros(batch, dtype=sim_specs["out"])

    for i, x in enumerate(H["x"]):
        H0["f_val"][i] = noisy_function(x)[H["f_ind"][i]]

    return H0, persis_info


def noisy_function(x):
    """ """
    x1 = x[0]
    x2 = x[1]
    term1 = (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2**2) * x2**2

    phi1 = 0.9 * sin(100 * norm(x, 1)) * cos(100 * norm(x, np.inf)) + 0.1 * cos(norm(x, 2))
    phi1 = phi1 * (4 * phi1**2 - 3)

    phi2 = 0.8 * sin(100 * norm(x, 1)) * cos(100 * norm(x, np.inf)) + 0.2 * cos(norm(x, 2))
    phi2 = phi2 * (4 * phi2**2 - 3)

    phi3 = 0.7 * sin(100 * norm(x, 1)) * cos(100 * norm(x, np.inf)) + 0.3 * cos(norm(x, 2))
    phi3 = phi3 * (4 * phi3**2 - 3)

    F = np.zeros(3)
    F[0] = (1 + 1e-1 * phi1) * term1
    F[1] = (1 + 1e-2 * phi2) * term2
    F[2] = (1 + 1e-3 * phi3) * term3

    return F
