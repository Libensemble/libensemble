"""
This module contains a periodic test function
"""

import numpy as np
from numpy import cos, sin


def func_wrapper(H, persis_info, sim_specs, libE_info):
    """
    Wraps an objective function
    """

    batch = len(H["x"])
    H0 = np.zeros(batch, dtype=sim_specs["out"])

    for i, x in enumerate(H["x"]):
        H0["f"][i] = periodic_func(x)

    return H0, persis_info


def periodic_func(x):
    """
    This function is periodic
    """
    return sin(x[0]) * cos(x[1])
